#!/usr/bin/env python
"""Collect paired clip-level audio / video / text embeddings for the projection fit.

For each VGGSound clip the script runs the Thinker's encoders and stores the mean
audio-encoder output, the mean video-encoder output and the mean input embedding
of the clip's caption (class label).

Manifest: JSONL with one clip per line, {"video": "clip.mp4", "caption": "class label"};
relative video paths resolve against --video-dir.

    python calibration/collect_embeddings.py --manifest /path/to/vggsound_train.jsonl \
        --video-dir /path/to/VGGSound/video --n 5000 --output data/embeddings_3b.pt

Output: {"audio", "video", "text": [N, D] float32, "mu_audio", "mu_video", "mu_text": [D]}.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import load_model  # noqa: E402

LONG_KEYS = {"input_ids", "attention_mask", "feature_attention_mask", "video_grid_thw",
             "image_grid_thw", "audio_feature_lengths"}
SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                 "perceiving auditory and visual inputs, as well as generating text and speech.")


def _first(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return getattr(x, "last_hidden_state", x)


def prepare_cpu(processor, video, caption):
    """Decode one clip and run the processor on CPU (runs inside a DataLoader worker)."""
    from qwen_omni_utils import process_mm_info
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video", "video": video},
                                         {"type": "text", "text": caption}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    return dict(processor(text=text, audio=audios, images=images, videos=videos,
                          return_tensors="pt", padding=True, use_audio_in_video=True))


def to_device(raw, model):
    enc_dtype = next(model.thinker.visual.parameters()).dtype
    out = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k in LONG_KEYS:
            out[k] = v.to(model.device, dtype=torch.long)
        elif k == "input_features":
            out[k] = v.to(model.device, dtype=torch.float32)
        elif k in ("pixel_values", "pixel_values_videos"):
            out[k] = v.to(model.device, dtype=enc_dtype)
        else:
            out[k] = v.to(model.device)
    return out


class Clips(Dataset):
    """Manifest rows -> processor inputs; a failed decode yields ``inputs=None``."""

    def __init__(self, rows, processor, offset=0):
        self.rows, self.processor, self.offset = rows, processor, offset

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        video, caption = self.rows[i]
        try:
            inputs, err = prepare_cpu(self.processor, video, caption), None
        except Exception as e:
            inputs, err = None, f"{type(e).__name__}: {str(e)[:100]}"
        return {"idx": self.offset + i, "video": video, "caption": caption, "inputs": inputs, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-Omni-3B",
                    choices=["Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--n", type=int, default=5000, help="number of successful clips")
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", default=None)
    # Use FlashAttention to limit encoder memory on full-resolution clips.
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--clip-timeout", type=int, default=30, help="seconds before a clip that hangs in decoding is skipped")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers that decode clips")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--encoders-only", action="store_true", help="release unused decoder layers after loading")
    args = ap.parse_args()

    model, processor, _ = load_model(args.model_name, cache_dir=args.cache_dir, attn_implementation=args.attn)
    thinker = model.thinker
    embed = thinker.get_input_embeddings()
    if args.encoders_only:
        thinker.model.layers = torch.nn.ModuleList()
        thinker.lm_head = torch.nn.Identity()
        torch.cuda.empty_cache()

    rows = []
    for row in (json.loads(l) for l in open(args.manifest) if l.strip()):
        caption = str(row.get("caption") or row.get("class_label") or "").strip()
        if row.get("video") and caption:
            video = row["video"] if Path(row["video"]).is_absolute() else str(Path(args.video_dir) / row["video"])
            rows.append((video, caption))

    assert 0 <= args.shard_index < args.num_shards
    global_indices = list(range(len(rows)))[args.shard_index::args.num_shards]
    rows = [rows[i] for i in global_indices]
    # Decode in workers so a stalled clip can be skipped after the timeout.
    A, V, T = [], [], []
    collected_indices, collected_paths = [], []
    pbar = tqdm(total=args.n, desc="clips")
    start = 0
    while start < len(rows) and len(A) < args.n:
        loader = DataLoader(Clips(rows[start:], processor, start), batch_size=None, num_workers=args.workers,
                            timeout=args.clip_timeout, prefetch_factor=1)
        last = start - 1
        try:
            for item in loader:
                if len(A) >= args.n:
                    break
                last = item["idx"]
                if item["inputs"] is None:
                    print(f"[skip] {item['video']}: {item['error']}")
                    continue
                inp = to_device(item["inputs"], model)
                if inp.get("pixel_values_videos") is None or inp.get("input_features") is None:
                    continue
                try:
                    with torch.no_grad():
                        vf = _first(thinker.get_video_features(inp["pixel_values_videos"], inp.get("video_grid_thw")))
                        af = _first(thinker.get_audio_features(
                            inp["input_features"], feature_attention_mask=inp.get("feature_attention_mask"),
                            audio_feature_lengths=inp.get("audio_feature_lengths")))
                        ids = processor.tokenizer.encode(item["caption"], add_special_tokens=False)
                        if not ids:
                            continue
                        te = embed(torch.tensor(ids, dtype=torch.long, device=embed.weight.device))
                except Exception as e:
                    print(f"[skip] {item['video']}: {type(e).__name__}: {str(e)[:100]}")
                    continue
                V.append(vf.reshape(-1, vf.shape[-1]).float().mean(0).cpu())
                A.append(af.reshape(-1, af.shape[-1]).float().mean(0).cpu())
                T.append(te.float().mean(0).cpu())
                collected_indices.append(global_indices[item["idx"]])
                collected_paths.append(item["video"])
                pbar.update(1)
        except RuntimeError as e:
            if "timed out" not in str(e).lower():
                raise
            print(f"[skip] {rows[last + 1][0]}: decode timed out after {args.clip_timeout}s")
            start = last + 2
            continue
        break
    pbar.close()

    if not A:
        raise RuntimeError("No embeddings collected")
    A, V, T = torch.stack(A), torch.stack(V), torch.stack(T)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"audio": A, "video": V, "text": T,
                "mu_audio": A.mean(0), "mu_video": V.mean(0), "mu_text": T.mean(0),
                "indices": collected_indices, "paths": collected_paths}, args.output)
    print(f"[done] {A.shape[0]} clips, D={A.shape[1]} -> {args.output}")


if __name__ == "__main__":
    main()
