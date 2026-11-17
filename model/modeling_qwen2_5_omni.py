"""Qwen2.5-Omni Thinker subclasses with ReMo prefill compression.

attach_remo replaces the loaded Thinker and text-model classes. Compression
runs after audio/video embeddings and TMRoPE positions are computed, before
the decoder layers. Without per-sample state, forward uses the base classes."""
from __future__ import annotations

from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)

from . import prune


class ReMoThinkerTextModel(Qwen2_5OmniThinkerTextModel):
    """Text model that compresses the prefill sequence before the decoder layers."""

    def forward(self, *args, **kwargs):
        names = ("input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds")
        kwargs.update(zip(names, args))
        return prune.inner_forward(self, super().forward, **kwargs)


class ReMoThinker(Qwen2_5OmniThinkerForConditionalGeneration):
    """Thinker that keeps ``input_ids`` so the text model can locate video/audio tokens."""

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        self.model._remo_input_ids = input_ids
        return super().forward(*args, **kwargs)


def attach_remo(model, state: dict) -> None:
    """Re-class ``model.thinker`` and ``model.thinker.model`` and store the ReMo state."""
    thinker, inner = model.thinker, model.thinker.model
    if not isinstance(inner, ReMoThinkerTextModel):
        inner.__class__ = ReMoThinkerTextModel
    if not isinstance(thinker, ReMoThinker):
        thinker.__class__ = ReMoThinker
    cfg = model.config.thinker_config
    state["video_token_id"] = getattr(cfg, "video_token_id", None) or getattr(cfg, "video_token_index", None)
    state["audio_token_id"] = getattr(cfg, "audio_token_id", None) or getattr(cfg, "audio_token_index", None)
    inner._remo = state
    inner._remo_sample = None
    inner._remo_input_ids = None
    inner._remo_last_stats = None
