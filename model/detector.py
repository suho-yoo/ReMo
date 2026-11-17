"""YOLO-seg object detection -> per-frame state for ReMo.

ReMo needs, for every video frame seen by the model (one temporal patch of
``temporal_patch`` sampled frames):

* ``labels``      : detected class names,
* ``seg_masks``   : union of the instance masks on a G x G grid,
* ``class_masks`` : one ``{"label", "grid"}`` entry per class (max over instances).

patch_state_live decodes the video and runs YOLO-seg. Frames use the Qwen-Omni
processor's sampling (smart_nframes + linspace) to align detections with video tokens.

The detector is ultralytics YOLO (AGPL-3.0). It is an external runtime dependency
and is not included in this repository.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Detector + frame sampling
# ---------------------------------------------------------------------------
def load_detector(weights: str = "yolov8n-seg.pt", device: str = "cuda"):
    from ultralytics import YOLO
    det = YOLO(weights)
    try:
        det.to(device)
    except Exception:
        pass
    return det


def sample_video_frames(video_path: str, yolo_fps: float = 2.0):
    """Decode RGB frames with the processor's sampling (``smart_nframes`` count,
    ``linspace`` indices). Returns (frames, indices, video_fps, total_frames)."""
    import decord
    from qwen_omni_utils.v2_5.vision_process import smart_nframes
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    video_fps = float(vr.get_avg_fps())
    nframes = smart_nframes({"fps": yolo_fps}, total_frames=total, video_fps=video_fps)
    idx = torch.linspace(0, total - 1, nframes).round().long().tolist()
    frames = vr.get_batch(idx).asnumpy()
    return [frames[i] for i in range(frames.shape[0])], idx, video_fps, total


def labels_from_result(result, conf_threshold: float) -> List[str]:
    if result.boxes is None or result.boxes.cls is None:
        return []
    confs = result.boxes.conf.detach().cpu().numpy()
    cls_idx = result.boxes.cls.detach().cpu().int().numpy()
    keep = confs >= conf_threshold
    names = result.names
    return sorted({names[int(c)] for c, k in zip(cls_idx, keep) if k})


def masks_from_result(result, conf_threshold: float, seg_grid: int):
    """Union mask and per-instance class masks, area-resized to seg_grid x seg_grid
    on the GPU. Returns (union_flat [G*G], [{"label", "flat"}])."""
    G = int(seg_grid)
    zero_flat = [0.0] * (G * G)
    if result.boxes is None or len(result.boxes) == 0:
        return zero_flat, []
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or masks.data.numel() == 0:
        return zero_flat, []
    keep = result.boxes.conf >= conf_threshold
    if not bool(keep.any()):
        return zero_flat, []
    md = masks.data[keep].float()
    grids = F.interpolate(md.unsqueeze(1), size=(G, G), mode="area").squeeze(1).clamp_(0.0, 1.0)
    union_flat = grids.amax(dim=0).reshape(-1).detach().cpu().tolist()
    cls_idx = result.boxes.cls[keep].int().detach().cpu().tolist()
    grids_cpu = grids.detach().cpu()
    per_class = [{"label": str(result.names[int(ci)]), "flat": grids_cpu[i].reshape(-1).tolist()}
                 for i, ci in enumerate(cls_idx)]
    return union_flat, per_class


def predict_entry(frames, frame_indices, video_fps: float, total_frames: int, detector, *,
                  yolo_fps: float, seg_grid: int, conf: float, batch_size: int) -> dict:
    """Run YOLO-seg on decoded frames; return per-frame detections."""
    labels, seg_flat, class_seg = [], [], []
    for i in range(0, len(frames), batch_size):
        for r in detector.predict(list(frames[i:i + batch_size]), conf=conf, verbose=False):
            labels.append(labels_from_result(r, conf))
            u_flat, pc = masks_from_result(r, conf, int(seg_grid))
            seg_flat.append(u_flat)
            class_seg.append(pc)
    return {
        "fps": yolo_fps, "video_fps": video_fps, "total_frames": int(total_frames),
        "nframes": len(frames), "frame_indices": frame_indices, "frame_labels": labels,
        "seg_grid_h": int(seg_grid), "seg_grid_w": int(seg_grid),
        "frame_seg_flat": seg_flat, "frame_class_seg": class_seg,
    }


# ---------------------------------------------------------------------------
# Per-frame detections -> temporal-patch state
# ---------------------------------------------------------------------------
def _subsample_frames(seq: list, model_nframes: int) -> list:
    n = len(seq)
    if n == model_nframes:
        return seq
    idx = torch.linspace(0, n - 1, model_nframes).round().long().tolist()
    return [seq[i] for i in idx]


def _model_nframes(entry: dict, processor_video_fps: Optional[float]) -> Optional[int]:
    total, video_fps = entry.get("total_frames"), entry.get("video_fps")
    if total is None or video_fps is None:
        return None
    try:
        from qwen_omni_utils.v2_5.vision_process import smart_nframes
        info = {} if processor_video_fps is None else {"fps": processor_video_fps}
        return int(smart_nframes(info, total_frames=int(total), video_fps=float(video_fps)))
    except Exception:
        return None


def _frames(entry: dict, key: str, model_nframes: Optional[int]) -> list:
    seq = list(entry[key])
    if model_nframes is not None and model_nframes < len(seq):
        seq = _subsample_frames(seq, model_nframes)
    return seq


def labels_per_patch(entry: dict, temporal_patch: int, model_nframes: Optional[int]) -> List[List[str]]:
    fl = _frames(entry, "frame_labels", model_nframes)
    out = []
    for k in range(0, len(fl), temporal_patch):
        merged = sorted(set(fl[k]))
        for j in range(1, temporal_patch):
            if k + j < len(fl):
                merged = sorted(set(merged).union(fl[k + j]))
        out.append(merged)
    return out


def masks_per_patch(entry: dict, temporal_patch: int, model_nframes: Optional[int]) -> List[np.ndarray]:
    if "frame_seg_flat" not in entry:
        return []
    gh, gw = int(entry["seg_grid_h"]), int(entry["seg_grid_w"])
    flat = _frames(entry, "frame_seg_flat", model_nframes)
    out = []
    for k in range(0, len(flat), temporal_patch):
        acc = None
        for j in range(temporal_patch):
            if k + j >= len(flat):
                break
            arr = np.asarray(flat[k + j], dtype=np.float32).reshape(gh, gw)
            acc = arr if acc is None else np.maximum(acc, arr)
        if acc is not None:
            out.append(acc)
    return out


def class_masks_per_patch(entry: dict, temporal_patch: int, model_nframes: Optional[int]) -> List[List[dict]]:
    if "frame_class_seg" not in entry:
        return []
    gh, gw = int(entry["seg_grid_h"]), int(entry["seg_grid_w"])
    flat = _frames(entry, "frame_class_seg", model_nframes)
    out = []
    for k in range(0, len(flat), temporal_patch):
        by_label: dict = {}
        for j in range(temporal_patch):
            if k + j >= len(flat):
                break
            for inst in flat[k + j]:
                lab = str(inst.get("label", "")).strip()
                if not lab:
                    continue
                arr = np.asarray(inst["flat"], dtype=np.float32).reshape(gh, gw)
                by_label[lab] = np.maximum(by_label[lab], arr) if lab in by_label else arr
        out.append([{"label": lab, "grid": grid} for lab, grid in sorted(by_label.items())])
    return out


def entry_to_patch_state(entry: dict, *, temporal_patch: int = 2,
                         processor_video_fps: Optional[float] = None
                         ) -> Tuple[List[List[str]], Optional[list], Optional[list]]:
    """(labels, seg_masks, class_masks) per model frame (temporal patch)."""
    mnf = _model_nframes(entry, processor_video_fps)
    labels = labels_per_patch(entry, temporal_patch, mnf)
    seg = masks_per_patch(entry, temporal_patch, mnf)
    cls = class_masks_per_patch(entry, temporal_patch, mnf)
    return labels, (seg or None), (cls or None)


def patch_state_live(video_path: str, detector, *, temporal_patch: int = 2,
                     processor_video_fps: Optional[float] = None, yolo_fps: float = 2.0,
                     seg_grid: int = 32, conf: float = 0.25, batch_size: int = 16):
    frames, idx, vfps, total = sample_video_frames(video_path, yolo_fps=yolo_fps)
    entry = predict_entry(frames, idx, vfps, total, detector, yolo_fps=yolo_fps,
                          seg_grid=seg_grid, conf=conf, batch_size=batch_size)
    return entry_to_patch_state(entry, temporal_patch=temporal_patch,
                                processor_video_fps=processor_video_fps)
