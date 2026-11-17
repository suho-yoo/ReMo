"""ReMo prefill compression before the Thinker's decoder layers.

Combines residual saliency, temporal merging, local detail restoration, and
class-name text proxies. Rebuilds embeddings, TMRoPE positions, and the
attention mask for the compressed sequence."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

Run = Tuple[int, int]


# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------
def find_runs(input_ids: torch.Tensor, token_id: int) -> List[Run]:
    """Contiguous [start, end) ranges where ``input_ids[0] == token_id``."""
    flat = input_ids[0]
    mask = (flat == token_id).to(torch.int32)
    if not mask.any():
        return []
    diff = mask[1:] - mask[:-1]
    rises = (diff == 1).nonzero(as_tuple=True)[0].tolist()
    falls = (diff == -1).nonzero(as_tuple=True)[0].tolist()
    if mask[0]:
        rises = [-1] + rises
    if mask[-1]:
        falls = falls + [mask.shape[0] - 1]
    return [(s + 1, e + 1) for s, e in zip(rises, falls)]


def split_into_frames(runs: List[Run], n_frames: int, n_spatial: int) -> Optional[List[Run]]:
    """Split the video-token runs into one run of ``n_spatial`` tokens per frame.

    With ``use_audio_in_video`` Qwen2.5-Omni interleaves video and audio in 2 s
    chunks, so each video run usually holds two frames. Returns None when the runs
    do not line up with the frame grid; the sample is then left unpruned.
    """
    if not runs:
        return None
    if n_frames != len(runs) and n_frames > len(runs) and n_frames % len(runs) == 0:
        k = n_frames // len(runs)
        if all((e - s) == k * n_spatial for s, e in runs):
            runs = [(s + i * n_spatial, s + (i + 1) * n_spatial) for s, e in runs for i in range(k)]
        else:
            return None
    frames: List[Run] = []
    for s, e in runs:
        if e - s <= 0:
            continue
        if (e - s) % n_spatial != 0:
            return None
        frames.extend((s + i * n_spatial, s + (i + 1) * n_spatial) for i in range((e - s) // n_spatial))
    return frames or runs


def _paired_audio_run(vs: int, ve: int, audio_runs: List[Run]) -> Optional[int]:
    """Index of the audio chunk paired with a video frame: the first audio run
    after the frame, else the closest one."""
    for idx, (a_s, _a_e) in enumerate(audio_runs):
        if a_s >= ve:
            return idx
    if not audio_runs:
        return None

    def _dist(ar: Run) -> int:
        a_s, a_e = ar
        if a_e <= vs:
            return vs - a_e
        if a_s >= ve:
            return a_s - ve
        return 0

    return min(range(len(audio_runs)), key=lambda i: _dist(audio_runs[i]))


# ---------------------------------------------------------------------------
# 1. Residual saliency
# ---------------------------------------------------------------------------
def audio_references(inputs_embeds: torch.Tensor, input_ids: torch.Tensor, frames: List[Run],
                     audio_token_id: Optional[int], proj: Dict[str, torch.Tensor]
                     ) -> List[Optional[torch.Tensor]]:
    """Mean audio embedding of the chunk paired with each frame, mapped to the
    common space: (a_t - mu_a) Pa. None when the sequence has no audio."""
    if audio_token_id is None:
        return [None] * len(frames)
    audio_runs = find_runs(input_ids, audio_token_id)
    if not audio_runs:
        return [None] * len(frames)
    refs: List[Optional[torch.Tensor]] = []
    for vs, ve in frames:
        aidx = _paired_audio_run(vs, ve, audio_runs)
        ref = None
        if aidx is not None:
            a_s, a_e = audio_runs[aidx]
            if a_e > a_s:
                ref = inputs_embeds[0, a_s:a_e].float().mean(dim=0, keepdim=True)
                mu_a = proj["mu_audio"].to(device=ref.device, dtype=ref.dtype)
                Pa = proj["Pa"].to(device=ref.device, dtype=ref.dtype)
                ref = (ref - mu_a.view(1, -1)) @ Pa
        refs.append(ref)
    return refs


def residual_saliency(inputs_embeds: torch.Tensor, frames: List[Run],
                      audio_refs: List[Optional[torch.Tensor]],
                      proj: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Per-frame saliency s = (1 - beta_t) s^vis + beta_t s^aud (float32, [n_spatial])."""
    chunks = torch.stack([inputs_embeds[0, s:e] for s, e in frames]).float()   # [T, N, D]
    frame_means = chunks.mean(dim=1, keepdim=True)
    c_norms = F.normalize(chunks, p=2, dim=-1)
    m_norms = F.normalize(frame_means, p=2, dim=-1).expand_as(c_norms)
    s_vis_all = 1.0 - F.cosine_similarity(c_norms, m_norms, dim=-1)           # [T, N]

    saliencies = []
    for t, (s, e) in enumerate(frames):
        s_vis = s_vis_all[t]
        a_ref = audio_refs[t] if t < len(audio_refs) else None
        if a_ref is None:
            saliencies.append(s_vis)
            continue
        a_norm = F.normalize(a_ref, p=2, dim=-1)
        Pv = proj["Pv"].to(device=s_vis.device, dtype=c_norms.dtype)
        mu_v = proj["mu_video"].to(device=s_vis.device, dtype=c_norms.dtype)
        v_can = F.normalize((inputs_embeds[0, s:e].to(c_norms.dtype) - mu_v.view(1, -1)) @ Pv, p=2, dim=-1)
        s_aud = 1.0 - F.cosine_similarity(v_can, a_norm.expand_as(v_can), dim=-1)
        Sv = float(s_vis.mean().item())
        Sa = float(s_aud.mean().item())
        beta = float(Sv / (Sv + Sa + 1e-8))
        saliencies.append((1.0 - beta) * s_vis + beta * s_aud)
    return saliencies


# ---------------------------------------------------------------------------
# 2. Temporal merging
# ---------------------------------------------------------------------------
def temporal_merge(inputs_embeds: torch.Tensor, keep_mask: torch.Tensor, frames: List[Run],
                   saliencies: List[torch.Tensor], tau: float, saliency_pct: float) -> int:
    """Merge redundant kept tokens of consecutive frames (in place).

    For positions kept in both frames t and t+1 with feature change
    1 - cos(v_t, v_{t+1}) < tau and min saliency at or below the ``saliency_pct``
    percentile of the pair, the token is replaced by the saliency-weighted average
    ``(1 - alpha) v_t + alpha v_{t+1}`` with alpha = s_{t+1} / (s_t + s_{t+1}),
    written to the frame with higher saliency; the other copy is dropped.
    Returns the number of merged positions.
    """
    merged = 0
    for i in range(len(frames) - 1):
        s_i, e_i = frames[i]
        s_j, e_j = frames[i + 1]
        shared_mask = keep_mask[s_i:e_i] & keep_mask[s_j:e_j]
        if not bool(shared_mask.any()):
            continue
        shared_idx = shared_mask.nonzero(as_tuple=False).squeeze(-1)
        z_i = inputs_embeds[0, s_i:e_i][shared_idx].float()
        z_j = inputs_embeds[0, s_j:e_j][shared_idx].float()
        gate = (1.0 - F.cosine_similarity(z_i, z_j, dim=-1)) < tau

        sal_i = saliencies[i][shared_idx].to(torch.float32)
        sal_j = saliencies[i + 1][shared_idx].to(torch.float32)
        sal_agg = torch.minimum(sal_i, sal_j)
        if sal_agg.numel() > 1:
            k_thr = max(1, int(round(sal_agg.numel() * saliency_pct)))
            thr = sal_agg.kthvalue(k_thr).values.item()
            gate = gate & (sal_agg <= thr)
        if not bool(gate.any()):
            continue
        fuse_local = shared_idx[gate]
        write_to_j = sal_j[gate] > sal_i[gate]

        z_i_g = inputs_embeds[0, s_i + fuse_local].float()
        z_j_g = inputs_embeds[0, s_j + fuse_local].float()
        sal_i_g = sal_i[gate].clamp_min(1e-8)
        sal_j_g = sal_j[gate].clamp_min(1e-8)
        alpha = (sal_j_g / (sal_i_g + sal_j_g)).unsqueeze(-1)
        fused = (1.0 - alpha) * z_i_g + alpha * z_j_g

        if bool(write_to_j.any()):
            idx_j = fuse_local[write_to_j]
            inputs_embeds[0, s_j + idx_j] = fused[write_to_j].to(inputs_embeds.dtype)
            keep_mask[s_i + idx_j] = False
        idx_i = fuse_local[~write_to_j]
        if idx_i.numel() > 0:
            inputs_embeds[0, s_i + idx_i] = fused[~write_to_j].to(inputs_embeds.dtype)
            keep_mask[s_j + idx_i] = False
        merged += int(fuse_local.numel())
    return merged


def compute_keep_mask(inputs_embeds: torch.Tensor, input_ids: torch.Tensor, cfg, proj,
                      *, video_grid_thw: torch.Tensor, merge_size: int,
                      video_token_id: int, audio_token_id: Optional[int]
                      ) -> Tuple[torch.Tensor, Optional[List[Run]], dict]:
    """Saliency selection + temporal merging. Returns (keep_mask [S], frames, stats).

    ``frames`` is None when the video tokens do not match the frame grid; the sample
    is then left unpruned. Note that temporal merging writes merged tokens into
    ``inputs_embeds`` in place.
    """
    S = input_ids.shape[1]
    keep_mask = torch.ones(S, dtype=torch.bool, device=inputs_embeds.device)
    runs = find_runs(input_ids, video_token_id)
    g = video_grid_thw
    thw = g[0] if g.dim() == 2 else g.flatten()[:3]
    T, H, W = int(thw[0]), int(thw[1]), int(thw[2])
    h, w = H // merge_size, W // merge_size
    n_spatial = h * w
    stats = {"video_tokens_total": sum(e - s for s, e in runs), "video_tokens_pruned": 0}

    frames = split_into_frames(runs, T, n_spatial)
    if frames is None:
        stats["mismatch"] = True
        return keep_mask, None, stats

    refs = audio_references(inputs_embeds, input_ids, frames, audio_token_id, proj)
    saliencies = residual_saliency(inputs_embeds, frames, refs, proj)

    n_keep = max(1, min(n_spatial, int(round(n_spatial * cfg.budget))))
    for (s, e), sal in zip(frames, saliencies):
        local = torch.zeros(e - s, dtype=torch.bool, device=keep_mask.device)
        local[sal.topk(min(n_keep, e - s), largest=True).indices] = True
        keep_mask[s:e] = local

    if len(frames) > 1:
        stats["merged_tokens"] = temporal_merge(
            inputs_embeds, keep_mask, frames, saliencies, cfg.merge_tau, cfg.merge_saliency_pct)
    stats["n_frames"] = len(frames)
    return keep_mask, frames, stats


# ---------------------------------------------------------------------------
# 3. Local detail restoration
# ---------------------------------------------------------------------------
def spatial_variation(embeds: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """d^l: sum of L2 feature differences to the horizontal/vertical neighbours."""
    E = embeds.reshape(h, w, -1).float()
    score = torch.zeros(h, w, device=embeds.device)
    diff_h = (E[:, 1:, :] - E[:, :-1, :]).norm(dim=-1)
    score[:, 1:] += diff_h
    score[:, :-1] += diff_h
    diff_v = (E[1:, :, :] - E[:-1, :, :]).norm(dim=-1)
    score[1:, :] += diff_v
    score[:-1, :] += diff_v
    return score.reshape(-1)


# ---------------------------------------------------------------------------
# 4. Text proxies
# ---------------------------------------------------------------------------
def class_mask_on_token_grid(item: dict, h: int, w: int) -> Optional[np.ndarray]:
    """Area-resample a YOLO class mask to the LLM token grid (h x w)."""
    grid = item.get("grid")
    if grid is None:
        flat = item.get("flat")
        if flat is None:
            return None
        arr = np.asarray(flat, dtype=np.float32).reshape(-1)
        if arr.size == h * w:
            return arr.reshape(h, w)
        g = int(round(arr.size ** 0.5))
        if g * g != arr.size:
            return None
        grid = arr.reshape(g, g)
    grid = np.asarray(grid, dtype=np.float32)
    if grid.shape == (h, w):
        return grid
    t = torch.from_numpy(np.ascontiguousarray(grid))[None, None]
    return F.interpolate(t, size=(h, w), mode="area")[0, 0].numpy()


def mask_centroid_position(class_masks: List[dict], class_name: str, position_ids: torch.Tensor,
                           start: int, end: int, h: int, w: int) -> Optional[torch.Tensor]:
    """TMRoPE position (t, h_i, w_i) of a class: mask-weighted mean of the frame's
    token positions (max over instances of the same class)."""
    if position_ids is None or end <= start or h <= 0 or w <= 0 or (end - start) != h * w:
        return None
    weights = None
    for item in class_masks or []:
        if str(item.get("label", "")).strip() != class_name:
            continue
        grid = class_mask_on_token_grid(item, h, w)
        if grid is None:
            continue
        wt = torch.as_tensor(grid, device=position_ids.device, dtype=torch.float32).reshape(-1)
        weights = wt if weights is None else torch.maximum(weights, wt)
    if weights is None or weights.numel() != (end - start):
        return None
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return None
    pp = position_ids[:, 0, start:end].float()
    return ((pp * weights.unsqueeze(0)).sum(dim=-1) / total).long()


def text_proxy_embeddings(labels: List[str], tokenizer, embed_layer, cache: dict
                          ) -> Tuple[Optional[torch.Tensor], List[str]]:
    """p_i = E_text(c_i): mean token embedding of each detected class name.
    Returns ([C, D], sorted class names)."""
    names = sorted({str(lab).strip() for lab in labels if str(lab).strip()})
    rows, kept = [], []
    for name in names:
        ids = cache.get(name)
        if ids is None:
            ids = tokenizer.encode(name, add_special_tokens=False)
            if len(cache) < 4096:
                cache[name] = ids
        if not ids:
            continue
        ids_t = torch.tensor(ids, device=embed_layer.weight.device, dtype=torch.long)
        rows.append(embed_layer(ids_t).mean(dim=0, keepdim=True))
        kept.append(name)
    if not rows:
        return None, []
    return torch.cat(rows, dim=0), kept


# ---------------------------------------------------------------------------
# Sequence rebuild
# ---------------------------------------------------------------------------
def rebuild_sequence(inputs_embeds: torch.Tensor, keep_mask: torch.Tensor, position_ids,
                     attention_mask, cache_position, frames: List[Run], cfg, *,
                     labels: List[List[str]], class_masks: Optional[List[List[dict]]],
                     h: int, w: int, tokenizer, embed_layer, token_cache: dict, stats: dict):
    """Rebuild the sequence frame by frame:
    [non-video tokens] [text proxies of frame t] [kept tokens of frame t] ...

    Local detail restoration runs here, per frame, before the proxies are placed.
    Returns (inputs_embeds, position_ids, attention_mask, cache_position).
    """
    S = inputs_embeds.shape[1]
    dev = inputs_embeds.device
    has_pos = position_ids is not None
    emb_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []
    n_proxies = 0

    def copy_span(p0: int, p1: int):
        if p1 <= p0:
            return
        emb_chunks.append(inputs_embeds[0, p0:p1])
        if has_pos:
            pos_chunks.append(position_ids[:, 0, p0:p1])

    prev_end = 0
    for t, (s, e) in enumerate(frames):
        copy_span(prev_end, s)
        frame_keep = keep_mask[s:e].clone()

        # Local detail restoration: restore the rho_l most spatially varying dropped tokens.
        if cfg.local_detail_ratio > 0.0:
            dropped = (~frame_keep).nonzero(as_tuple=False).squeeze(-1)
            n_drop = int(dropped.shape[0])
            if n_drop > 1:
                d_local = spatial_variation(inputs_embeds[0, s:e], h, w)[dropped]
                n_restore = max(1, int(round(n_drop * cfg.local_detail_ratio)))
                frame_keep[dropped[d_local.topk(n_restore, largest=True).indices]] = True
            keep_mask[s:e] = frame_keep

        # Text proxies for the classes detected in this frame.
        frame_labels = labels[t] if t < len(labels) else []
        if frame_labels and bool((~frame_keep).any()):
            proxies, names = text_proxy_embeddings(frame_labels, tokenizer, embed_layer, token_cache)
            if proxies is not None:
                frame_masks = class_masks[t] if class_masks and t < len(class_masks) else None
                if frame_masks is None or (e - s) != h * w:
                    raise RuntimeError(
                        "ReMo places text proxies at the YOLO class-mask centroid, but class "
                        f"masks are missing for frame {t} (h={h}, w={w}, tokens={e - s}).")
                frame_center = position_ids[:, 0, s:e].float().mean(dim=-1).long() if has_pos else None
                for ci, name in enumerate(names):
                    emb_chunks.append(proxies[ci:ci + 1].to(inputs_embeds.dtype))
                    n_proxies += 1
                    if has_pos:
                        pos = mask_centroid_position(frame_masks, name, position_ids, s, e, h, w)
                        pos_chunks.append((pos if pos is not None else frame_center).unsqueeze(-1))

        if frame_keep.any():
            emb_chunks.append(inputs_embeds[0, s:e][frame_keep])
            if has_pos:
                pos_chunks.append(position_ids[:, 0, s:e][:, frame_keep])
        prev_end = e
    copy_span(prev_end, S)

    stats["text_proxy_tokens"] = n_proxies
    new_ie = torch.cat(emb_chunks, dim=0).unsqueeze(0).to(dev)
    new_S = new_ie.shape[1]
    new_pos = torch.cat(pos_chunks, dim=-1).unsqueeze(1).to(dev) if has_pos and pos_chunks else None
    new_attn = (torch.ones(1, new_S, dtype=attention_mask.dtype, device=dev)
                if attention_mask is not None else None)
    new_cp = (torch.arange(new_S, device=dev, dtype=cache_position.dtype)
              if cache_position is not None else None)
    return new_ie, new_pos, new_attn, new_cp


# ---------------------------------------------------------------------------
# Inner text-model forward
# ---------------------------------------------------------------------------
def inner_forward(self, stock_forward: Callable, input_ids=None, attention_mask=None,
                  position_ids=None, past_key_values=None, inputs_embeds=None,
                  cache_position=None, **kwargs):
    """Text-model forward with ReMo applied on the prefill.

    ``self`` is the Thinker's inner text model. ``self._remo`` holds the attached
    controller state (config, projection, tokenizer, token ids) and
    ``self._remo_sample`` the per-sample YOLO state; without them this is the stock
    forward.
    """
    remo = getattr(self, "_remo", None)
    sample = getattr(self, "_remo_sample", None)
    last_ids = getattr(self, "_remo_input_ids", None)

    is_prefill = (
        remo is not None
        and sample is not None
        and sample.get("labels") is not None
        and inputs_embeds is not None
        and inputs_embeds.shape[1] > 1
        and last_ids is not None
        and last_ids.shape[1] == inputs_embeds.shape[1]
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if is_prefill:
        cfg = remo["config"]
        proj = remo["projection"]
        stats: dict = {}
        if not sample.get("seg_masks") or sample.get("video_grid_thw") is None or not sample.get("merge_size"):
            frames = None
            runs = find_runs(last_ids, remo["video_token_id"])
            stats = {"video_tokens_total": sum(e - s for s, e in runs), "video_tokens_pruned": 0}
        else:
            keep_mask, frames, stats = compute_keep_mask(
                inputs_embeds, last_ids, cfg, proj,
                video_grid_thw=sample["video_grid_thw"], merge_size=int(sample["merge_size"]),
                video_token_id=remo["video_token_id"], audio_token_id=remo["audio_token_id"])

        if frames is None:
            print(f"[ReMo] video tokens do not match the frame grid for sample "
                  f"{sample.get('sample_id', '?')}; running it unpruned", flush=True)
        else:
            g = sample["video_grid_thw"]
            thw = g[0] if g.dim() == 2 else g.flatten()[:3]
            ms = int(sample["merge_size"])
            h, w = int(thw[1]) // ms, int(thw[2]) // ms
            inputs_embeds, position_ids, attention_mask, cache_position = rebuild_sequence(
                inputs_embeds, keep_mask, position_ids, attention_mask, cache_position, frames, cfg,
                labels=sample["labels"], class_masks=sample.get("class_masks"), h=h, w=w,
                tokenizer=remo["tokenizer"], embed_layer=self.get_input_embeddings(),
                token_cache=remo["token_cache"], stats=stats)
            input_ids = None

            total = sum(e - s for s, e in frames)
            kept = sum(int(keep_mask[s:e].sum().item()) for s, e in frames)
            stats["video_tokens_total"] = total
            stats["video_tokens_pruned"] = total - kept
            stats["new_seq_len"] = int(inputs_embeds.shape[1])
            if cfg.verbose:
                print(f"[ReMo] {sample.get('sample_id', '?')}: video {total} -> {kept}, "
                      f"proxies {stats.get('text_proxy_tokens', 0)}, seq -> {stats['new_seq_len']}",
                      flush=True)
        self._remo_last_stats = stats

    return stock_forward(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        cache_position=cache_position, **kwargs)
