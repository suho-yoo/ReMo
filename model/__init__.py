import os

import torch

from .remo import ReMo, ReMoConfig, default_projection


def load_model(model_name="Qwen/Qwen2.5-Omni-3B", use_remo=False, cache_dir=None,
               attn_implementation="flash_attention_2", **remo_overrides):
    """Load Qwen2.5-Omni (talker disabled). With ``use_remo=True`` ReMo is attached
    with the default configuration. Returns (model, processor, remo or None)."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    cache_dir = cache_dir or os.environ.get("MODEL_CACHE_DIR")
    model_id = model_name
    if not os.path.isdir(model_name):
        # Resolve the snapshot directory first: Qwen2.5-Omni loads its speaker file
        # (spk_dict.pt) without forwarding cache_dir, which fails offline otherwise.
        from huggingface_hub import snapshot_download
        model_name = snapshot_download(model_name, cache_dir=cache_dir)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    remo = None
    if use_remo:
        overrides = {k: v for k, v in remo_overrides.items() if v is not None}
        overrides.setdefault("projection", default_projection(model_id))
        remo = ReMo(**overrides)
        remo.attach(model, processor)
    return model, processor, remo


__all__ = ["load_model", "ReMo", "ReMoConfig"]
