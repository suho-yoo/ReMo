# ReMo model

The loader returns the original Transformers Qwen2.5-Omni model for baseline
inference. With `use_remo=True` (`--remo`), `modeling_qwen2_5_omni.py` swaps the
Thinker and its text model for subclasses that compress the prefill sequence
before the decoder layers, using the paper configuration in `remo.py`.

| File | Paper |
|---|---|
| `prune.py` `residual_saliency`, `audio_references` | Sec. 4.1 residual saliency (s^vis, s^aud, beta_t, rho_av) |
| `projection.py` | Sec. 4.1 cross-modal projection to a common space with regularized CCA (Eq. 4-6) |
| `prune.py` `text_proxy_embeddings`, `mask_centroid_position` | Sec. 4.2 semantic offloading to text proxies, pos(p_i) = (t, h_i, w_i) |
| `prune.py` `spatial_variation`, `rebuild_sequence` | Sec. 4.3 local detail restoration (d^l, rho_l) |
| `prune.py` `temporal_merge` | Sec. 4.3 temporal merging (tau) |
| `detector.py` | YOLOv8-seg labels and class masks per frame |
| `remo.py` | configuration and per-sample state |

The integration depends on Transformers 4.52.3 internals.
