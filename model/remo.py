"""ReMo configuration, projection loading, and per-sample state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .modeling_qwen2_5_omni import attach_remo
from .projection import load_projection

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ReMoConfig:
    """ReMo compression settings."""
    # Common-space projection (calibration/<size>/projection.pt).
    projection: Optional[str] = None
    # Residual saliency (Sec. 4.1): fraction of tokens kept per frame, rho_av.
    budget: float = 0.40
    # Local detail restoration (Sec. 4.3): fraction of a frame's dropped tokens
    # restored by spatial variation, rho_l.
    local_detail_ratio: float = 0.05
    # Temporal merging (Sec. 4.3): feature-change threshold tau and the saliency
    # percentile below which a position may be merged.
    merge_tau: float = 0.05
    merge_saliency_pct: float = 0.5
    verbose: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def default_projection(model_name: str) -> str:
    size = "7b" if "7b" in model_name.lower() else "3b"
    return str(ROOT / "calibration" / size / "projection.pt")


class ReMo:
    """Holds the configuration and the projection, and sets the per-sample state."""

    def __init__(self, config: Optional[ReMoConfig] = None, **overrides):
        self.config = config or ReMoConfig()
        for k, v in overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"ReMoConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._projection = None

    @property
    def projection(self):
        if self._projection is None:
            self._projection = load_projection(self.config.projection)
        return self._projection

    def attach(self, model, processor) -> None:
        attach_remo(model, {
            "config": self.config,
            "projection": self.projection,
            "tokenizer": processor.tokenizer,
            "token_cache": {},
        })

    def set_sample_state(self, model, *, labels: Optional[List[List[str]]],
                         seg_masks: Optional[list] = None,
                         class_masks: Optional[List[List[dict]]] = None,
                         video_grid_thw: Optional[torch.Tensor] = None,
                         merge_size: Optional[int] = None,
                         sample_id: Optional[str] = None) -> None:
        """YOLO state per video frame: class names, union mask, ``{"label", "grid"}``
        class masks. The next prefill is compressed."""
        inner = model.thinker.model
        inner._remo_sample = {
            "labels": labels, "seg_masks": seg_masks, "class_masks": class_masks,
            "video_grid_thw": video_grid_thw, "merge_size": merge_size, "sample_id": sample_id,
        }
        inner._remo_last_stats = None

    def clear_state(self, model) -> None:
        model.thinker.model._remo_sample = None

    def last_stats(self, model) -> Optional[dict]:
        """Token counts of the last compressed prefill."""
        return getattr(model.thinker.model, "_remo_last_stats", None)
