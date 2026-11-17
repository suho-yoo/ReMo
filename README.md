# Out of Sight, Still in Mind: Token Compression for Omni-LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2607.21179-b31b1b.svg)](https://arxiv.org/abs/2607.21179)

Official implementation of **ReMo**, a training-free token compression method for Omni-LLMs.
ReMo keeps the video tokens that neither the frame nor the paired audio explains, moves the
semantics of dropped object regions into text proxies placed at the objects' positions,
restores a few tokens with local detail, and merges tokens that do not change over time.

This release includes WorldSense and Daily-Omni inference. The default model is
Qwen2.5-Omni-3B. A single `--remo` flag enables ReMo with the paper configuration.
Without the flag, inference uses the original Qwen2.5-Omni baseline.

## Installation

Run commands from the repository root.

```bash
conda create -n remo python=3.10 -y
conda activate remo
conda install -c conda-forge ffmpeg -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install packaging ninja
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

Object masks come from YOLOv8n-seg ([Ultralytics](https://github.com/ultralytics/ultralytics),
AGPL-3.0, installed through `requirements.txt`). The weights are downloaded on first use.

## WorldSense

### Data

Download the videos from [WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense).
Set `--video-dir` to the directory containing the `.mp4` files with audio.
The included [json/worldsense.json](json/worldsense.json) contains 799 questions with video
filenames relative to this directory.

```text
/path/to/WorldSense/
  KZsaltBw.mp4
  ...
```

### Inference

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --video-dir /path/to/WorldSense
```

ReMo:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --remo \
  --video-dir /path/to/WorldSense
```

Both modes use the same prompt, audio/video inputs, and greedy decoding.
Model weights are downloaded from Hugging Face.

Predictions are saved to `outputs/worldsense_3b_baseline.jsonl` and
`outputs/worldsense_3b_remo.jsonl`. With `--remo`, each row also records the number
of removed video tokens and inserted text proxies.

### Evaluation

```bash
python eval/eval_worldsense.py outputs/worldsense_3b_baseline.jsonl
python eval/eval_worldsense.py outputs/worldsense_3b_remo.jsonl
```

A complete run reports `evaluated: 799` and `missing: 0`.
The evaluator reports accuracy in percent and counts invalid responses as incorrect.

## Daily-Omni

### Data

Download `Videos.tar` from the official [Daily-Omni dataset](https://huggingface.co/datasets/liarliar/Daily-Omni/tree/main)
and extract it under your Daily-Omni directory. See the
[official repository](https://github.com/Lliar-liar/Daily-Omni) for benchmark details.

The included [json/dailyomni.json](json/dailyomni.json) contains 1,197 questions. Set
`--video-dir` to the Daily-Omni root containing `Videos/<video_id>/<video_id>_video.mp4`.

```text
/path/to/Daily-Omni/
  Videos/
    <video_id>/
      <video_id>_video.mp4
```

### Inference

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_dailyomni.py \
  --video-dir /path/to/Daily-Omni
```

ReMo:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_dailyomni.py \
  --remo \
  --video-dir /path/to/Daily-Omni
```

Predictions are saved to `outputs/dailyomni_3b_baseline.jsonl` and
`outputs/dailyomni_3b_remo.jsonl`.

### Evaluation

```bash
python eval/eval_dailyomni.py outputs/dailyomni_3b_baseline.jsonl
python eval/eval_dailyomni.py outputs/dailyomni_3b_remo.jsonl
```

A complete run reports `evaluated: 1197` and `missing: 0`.

## Calibration

The supplied projection in `calibration/3b/` is ready for
inference and is loaded automatically with `--remo`. It is a weighted, regularized
CCA projection (Eq. 4-6; eps = 0.05, K = 512) fitted on CPU from paired VGGSound
train-clip embeddings: 5,000 pairs for 3B. To regenerate a
projection, collect paired embeddings and fit:

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/collect_embeddings.py \
  --manifest /path/to/vggsound_train.jsonl \
  --video-dir /path/to/VGGSound/video \
  --n 5000 --output data/embeddings_3b.pt

python calibration/fit_projection.py \
  --embeddings data/embeddings_3b.pt \
  --output calibration/3b/new_projection.pt
```

Run inference with `--projection calibration/3b/new_projection.pt`.
`calibration/eval_retrieval.py` reports the audio-to-video retrieval of a projection
on held-out embeddings (Tab. 5). See [calibration/README.md](calibration/README.md)
for the manifest format and the retrieval numbers of the supplied files.

## Code structure

```text
inference/
  inference_worldsense.py     WorldSense entry point
  inference_dailyomni.py      Daily-Omni entry point
  run.py                      shared loop: YOLO, ReMo state, generation
  data.py                     question loading and answer parsing
  inputs.py                   prompt and audio/video preprocessing
model/
  __init__.py                 baseline / ReMo model loader
  modeling_qwen2_5_omni.py    Qwen2.5-Omni Thinker with ReMo integration
  remo.py                     configuration and per-sample state
  prune.py                    residual saliency, text proxies, detail restoration, merging
  projection.py               regularized CCA common space
  detector.py                 YOLOv8-seg labels and class masks per frame
eval/
  eval_worldsense.py          WorldSense prediction scoring
  eval_dailyomni.py           Daily-Omni prediction scoring
calibration/
  collect_embeddings.py       paired VGGSound audio/video/text embeddings
  fit_projection.py           CCA projection fit
  eval_retrieval.py           audio-to-video retrieval check (Tab. 5)
  3b/                         shipped 3B projection
json/
  worldsense.json             799 multiple-choice questions
  dailyomni.json              1,197 multiple-choice questions
```

## Acknowledgments

This implementation builds on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) through
[Transformers](https://github.com/huggingface/transformers) and uses
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object masks.
The projection is fitted on [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

## Citation

```bibtex
@article{yoo2026out,
  title={Out of Sight, Still in Mind: Token Compression for Omni-LLMs},
  author={Yoo, Suho and Jang, Youngjoon and Cho, Hyebin and Chung, Joon Son},
  journal={arXiv preprint arXiv:2607.21179},
  year={2026}
}
```

## License

The code is released under the Apache-2.0 license (see [LICENSE](LICENSE)).
Third-party attributions are listed in [NOTICE](NOTICE). Ultralytics YOLO is an external
AGPL-3.0 dependency. Model weights and datasets keep their original licenses.
