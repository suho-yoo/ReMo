# Calibration

ReMo has no trained weights. Its only fitted component is the common-space
projection (Sec. 4.1, Eq. 4-6): a weighted, regularized CCA between clip-level
audio and video embeddings of the Thinker. Each covariance is shrunk towards a
scaled identity with eps = 0.05 before whitening, and the top K = 512 canonical
directions are kept, scaled by the square root of their canonical correlations.

`3b/projection.pt` is loaded automatically with `--remo` for Qwen2.5-Omni-3B.
It was fitted on CPU (float32 inputs, float64 eigendecomposition) from paired
VGGSound train-clip embeddings: 5,000 pairs. The file stores `mu_audio`,
`mu_video` (the embedding means used for centering), `mu_text`, `Pa`, `Pv` and the
fit settings under `metadata`.

## Retrieval check

Audio-to-video retrieval on held-out VGGSound test embeddings, one pair per class
(309 classes), averaged over sampling seeds 0-49. The supplied projection has
the following reference results:

| Model | Space | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|
| Qwen2.5-Omni-3B | Raw | 2.8 | 10.2 | 16.3 | 64 |
| | Projected | 30.6 | 54.2 | 62.7 | 4 |

```bash
python calibration/eval_retrieval.py \
  --embeddings data/vggsound_test_3b.pt \
  --projection calibration/3b/projection.pt
```

## Regenerate a projection

1. Collect paired embeddings. The manifest is JSONL with one clip per line,
   `{"video": "clip.mp4", "caption": "class label"}`; only the encoders and the
   embedding table are run.

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/collect_embeddings.py \
  --manifest /path/to/vggsound_train.jsonl \
  --video-dir /path/to/VGGSound/video \
  --n 5000 --output data/embeddings_3b.pt
```

2. Fit the projection:

```bash
python calibration/fit_projection.py \
  --embeddings data/embeddings_3b.pt \
  --output calibration/3b/new_projection.pt
```

Use the new file with `--projection calibration/3b/new_projection.pt`
at inference.
