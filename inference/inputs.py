import torch
from qwen_omni_utils import process_mm_info

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    " Please analyze the video carefully and select the most appropriate answer from the given options."
)
LONG_KEYS = frozenset({
    "input_ids", "attention_mask", "video_grid_thw", "image_grid_thw",
    "feature_attention_mask", "position_ids", "cache_position", "rope_deltas",
    "labels", "audio_feature_lengths",
})


def build_inputs(processor, model, video, prompt, *, max_pixels=None, fps=None):
    """Chat template + audio/video preprocessing (audio taken from the video track)."""
    vid = {"type": "video", "video": str(video)}
    if max_pixels is not None:
        vid["max_pixels"] = int(max_pixels)
    if fps is not None:
        vid["fps"] = fps
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [vid, {"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, _, videos = process_mm_info(conversation, use_audio_in_video=True)
    raw = processor(text=text, audio=audios, images=None, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            inputs[key] = value
        elif key in LONG_KEYS:
            inputs[key] = value.to(model.device, dtype=torch.long)
        elif key == "input_features":
            inputs[key] = value.to(model.device, dtype=torch.float32)
        elif key in ("pixel_values", "pixel_values_videos"):
            inputs[key] = value.to(model.device).to(model.dtype)
        else:
            inputs[key] = value.to(model.device)
    return inputs
