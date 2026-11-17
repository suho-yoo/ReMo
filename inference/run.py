"""Shared inference loop for WorldSense and Daily-Omni."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.data import load_items, parse_choice, video_path  # noqa: E402
from inference.inputs import build_inputs  # noqa: E402
from model import load_model  # noqa: E402
from model.detector import load_detector, patch_state_live  # noqa: E402


def output_file(args, bench):
    if args.output:
        return Path(args.output)
    size = args.model_name.rsplit("-", 1)[-1].lower()
    return Path(args.output_dir) / f"{bench}_{size}_{'remo' if args.remo else 'baseline'}.jsonl"


def yolo_state(args, detector, path):
    return patch_state_live(str(path), detector, temporal_patch=args.temporal_patch,
                            processor_video_fps=args.video_fps, yolo_fps=args.yolo_fps,
                            seg_grid=args.yolo_seg_grid, conf=args.yolo_conf,
                            batch_size=args.yolo_batch_size)


def run(args, bench):
    items = load_items(args.question_file)
    if args.max_samples:
        items = items[: args.max_samples]
    out_path = output_file(args, bench)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor, remo = load_model(
        args.model_name, use_remo=args.remo, cache_dir=args.cache_dir, attn_implementation=args.attn,
        projection=args.projection, budget=args.budget, local_detail_ratio=args.local_detail_ratio)
    merge_size = int(processor.video_processor.merge_size)
    detector = load_detector(args.yolo_weights) if remo is not None else None
    print(f"Model: {args.model_name}; mode: {'ReMo' if remo else 'baseline'}")
    print(f"Questions: {len(items)}; output: {out_path}")

    # Skip completed sample IDs; discard incomplete JSON rows.
    done = set()
    if out_path.exists():
        rows = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        done = {str(r["id"]) for r in rows}

    correct = total = pruned = seq = 0
    t0 = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as handle:
        for item in tqdm(items):
            if item["id"] in done:
                continue
            path = video_path(item, args.video_dir)
            if remo is not None:
                try:
                    labels, seg, cls = yolo_state(args, detector, path)
                except Exception as exc:
                    print(f"[warn] YOLO failed on {path}: {exc}; running this sample unpruned")
                    labels, seg, cls = None, None, None
            inputs = build_inputs(processor, model, path, item["prompt"],
                                  max_pixels=args.video_max_pixels, fps=args.video_fps)
            prompt_len = int(inputs["input_ids"].shape[1])
            stats = None
            if remo is not None:
                remo.set_sample_state(model, labels=labels, seg_masks=seg, class_masks=cls,
                                      video_grid_thw=inputs.get("video_grid_thw"),
                                      merge_size=merge_size, sample_id=item["id"])
            try:
                with torch.no_grad():
                    out_ids = model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                                             thinker_max_new_tokens=args.max_new_tokens)
                if remo is not None:
                    stats = remo.last_stats(model)
            finally:
                if remo is not None:
                    remo.clear_state(model)
            response = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            pred = parse_choice(response)
            ok = pred == item["answer"]
            vp = int(stats.get("video_tokens_pruned", 0)) if stats else 0
            correct += int(ok)
            total += 1
            pruned += vp
            seq += prompt_len
            handle.write(json.dumps({
                "id": item["id"], "video": item["video"], "answer": item["answer"],
                "response": response, "pred": pred, "is_correct": ok,
                "prompt_len": prompt_len, "video_tokens_pruned": vp,
                "text_proxy_tokens": int(stats.get("text_proxy_tokens", 0)) if stats else 0,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    if total:
        print(f"Accuracy (this run): {100 * correct / total:.2f} ({correct}/{total}); "
              f"removed tokens: {100 * pruned / max(1, seq):.1f}% of the input; "
              f"{time.perf_counter() - t0:.0f}s")


def parse_args(bench, default_video_dir):
    parser = argparse.ArgumentParser(description=f"{bench} inference (baseline or ReMo)")
    parser.add_argument("--remo", action="store_true", help="enable ReMo (paper configuration)")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Omni-3B",
                        choices=["Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B"])
    parser.add_argument("--question-file", default=str(ROOT / f"json/{bench}.json"))
    parser.add_argument("--video-dir", default=default_video_dir)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output", default=None, help="output JSONL (default: outputs/<bench>_<size>_<mode>.jsonl)")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache dir")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    # ReMo parameters
    parser.add_argument("--projection", default=None, help="default: calibration/<size>/projection.pt")
    parser.add_argument("--budget", type=float, default=None, help="rho_av (default 0.40)")
    parser.add_argument("--local-detail-ratio", type=float, default=None, help="rho_l (default 0.05)")
    # video / YOLO
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--temporal-patch", type=int, default=2)
    parser.add_argument("--yolo-weights", default="yolov8n-seg.pt")
    parser.add_argument("--yolo-fps", type=float, default=2.0)
    parser.add_argument("--yolo-seg-grid", type=int, default=32)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    return parser.parse_args()
