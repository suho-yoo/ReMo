import json
import re
from pathlib import Path


def parse_choice(text):
    """First A-D choice letter in a free-form answer (None if absent)."""
    text = (text or "").strip()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-D])\s*[.)\::-]", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-D])", text.upper())
    return match.group(1) if match else None


def load_items(path):
    """Benchmark JSON -> items with ``id``, ``video``, ``prompt`` and ``answer``."""
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    seen = set()
    for index, item in enumerate(items):
        item_id = str(item.get("id", index))
        if item_id in seen:
            raise ValueError(f"Duplicate sample id: {item_id}")
        seen.add(item_id)
        item["id"] = item_id
        turns = {turn["from"]: turn["value"] for turn in item["conversations"]}
        item["prompt"] = turns["human"].replace("<image>", "").replace("<video>", "").strip()
        item["answer"] = parse_choice(turns["gpt"]) or turns["gpt"].strip().upper()[:1]
    return items


def video_path(item, video_dir):
    path = Path(item["video"])
    return path if path.is_absolute() else Path(video_dir) / path
