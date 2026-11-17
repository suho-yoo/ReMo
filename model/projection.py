"""Cross-modal projection with weighted, regularized CCA.

Covariances use shrinkage (1 - eps) C + eps * mean(diag C) I before whitening.
The top K canonical directions are weighted by the square roots of their
canonical correlations. Audio and video map to the common space as
(a - mu_a) Pa and (v - mu_v) Pv.

calibration/fit_projection.py fits and saves the projection for inference."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

PROJECTION_KEYS = ("mu_audio", "mu_video", "mu_text", "Pa", "Pv")


def _sym_inv_sqrt(C: torch.Tensor, eps: float, use_cuda: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Regularized C^{-1/2} and C^{1/2} of a covariance matrix (Eq. 4).

    C is shrunk towards a scaled identity, (1 - eps) C + eps * mean(diag C) I, before
    the eigendecomposition. eigh runs on CUDA when available (and ``use_cuda``); the
    result is returned on the input device.
    """
    dev0 = C.device
    if use_cuda and torch.cuda.is_available() and C.device.type != "cuda":
        Cg = C.to("cuda")
    else:
        Cg = C
    d = Cg.shape[0]
    Cg = Cg.double()  # float64 eigh: stable, and avoids an MKL ssyevd failure in torch 2.6
    Cg = (1 - eps) * Cg + eps * Cg.diagonal().mean() * torch.eye(d, device=Cg.device, dtype=Cg.dtype)
    evals, evecs = torch.linalg.eigh(Cg)
    evals = evals.clamp_min(1e-8)
    inv_sqrt = ((evecs * evals.rsqrt()) @ evecs.T).to(device=dev0, dtype=C.dtype)
    sqrt = ((evecs * evals.sqrt()) @ evecs.T).to(device=dev0, dtype=C.dtype)
    return inv_sqrt, sqrt


def fit_cca_projectors(
    audio: torch.Tensor,
    video: torch.Tensor,
    mu_audio: torch.Tensor,
    mu_video: torch.Tensor,
    *,
    d_keep: int = 512,
    eps: float = 0.05,
    weighted: bool = True,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit (Pa, Pv) from paired audio/video embeddings [N, D] (weighted CCA).

    ``device`` defaults to CUDA when available. Returns float32 CPU tensors [D, d_keep].
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev != "cpu"
    s_samp = audio.float().to(dev)
    m_samp = video.float().to(dev)
    n = min(s_samp.shape[0], m_samp.shape[0])
    ds = s_samp[:n] - mu_audio.float().to(dev)
    dm = m_samp[:n] - mu_video.float().to(dev)
    Ca_inv, _ = _sym_inv_sqrt(ds.T @ ds / n, eps, use_cuda)
    Cv_inv, _ = _sym_inv_sqrt(dm.T @ dm / n, eps, use_cuda)
    T = (ds @ Ca_inv).T @ (dm @ Cv_inv) / n
    U, S, Vt = torch.linalg.svd(T)
    d_keep = min(d_keep, U.shape[1])
    Pa = (Ca_inv @ U[:, :d_keep]).to(torch.float32).cpu()
    Pv = (Cv_inv @ Vt[:d_keep].T).to(torch.float32).cpu()
    S = S[:d_keep].to(torch.float32).cpu()
    if weighted:
        w = S.sqrt().clamp_min(1e-6)
        Pa = Pa * w.view(1, -1)
        Pv = Pv * w.view(1, -1)
    return Pa, Pv


def load_projection(path: Optional[str]) -> Dict[str, torch.Tensor]:
    """Load a projection file written by ``calibration/fit_projection.py``."""
    if not path:
        raise ValueError(
            "ReMo needs a common-space projection file; see calibration/README.md.")
    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in PROJECTION_KEYS if data.get(k) is None]
    if missing:
        raise KeyError(f"{path} is missing {missing}; refit it with calibration/fit_projection.py")
    return {k: data[k] for k in PROJECTION_KEYS}
