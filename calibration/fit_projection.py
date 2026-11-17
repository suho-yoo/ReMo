"""Fit the CCA common-space projection (Sec. 4.1, Eq. 4-6) used by ReMo.

    python calibration/fit_projection.py --embeddings /path/to/embeddings_3b.pt \
        --output calibration/3b/new_projection.pt

The embeddings are centred by their own means (mu_audio, mu_video), whitened with the
regularized covariance (eps = 0.05), and the top K = 512 canonical directions are kept,
scaled by sqrt(sigma_k).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.projection import fit_cca_projectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="output of calibration/collect_embeddings.py")
    ap.add_argument("--d-keep", type=int, default=512, help="number of canonical directions K")
    ap.add_argument("--eps", type=float, default=0.05, help="covariance regularizer (shrinkage)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (fixed for reproducibility)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    emb = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    n = int(min(emb["audio"].shape[0], emb["video"].shape[0]))
    audio, video = emb["audio"][:n].float(), emb["video"][:n].float()
    mu = {"mu_audio": audio.mean(0), "mu_video": video.mean(0)}
    mu["mu_text"] = emb["text"][:n].float().mean(0) if "text" in emb else torch.zeros_like(mu["mu_audio"])
    Pa, Pv = fit_cca_projectors(audio, video, mu["mu_audio"], mu["mu_video"],
                                d_keep=args.d_keep, eps=args.eps, weighted=True, device=args.device)
    out = {
        "mu_audio": mu["mu_audio"], "mu_video": mu["mu_video"], "mu_text": mu["mu_text"],
        "Pa": Pa, "Pv": Pv,
        "metadata": {
            "d_keep": int(Pa.shape[1]), "eps": args.eps, "regularizer": "shrinkage", "weighted": True,
            "n_pairs": n, "device": args.device, "torch_threads": args.threads,
            "torch": str(torch.__version__), "dtype": "float32",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Pa {tuple(Pa.shape)} Pv {tuple(Pv.shape)} from {n} pairs -> {args.output}")


if __name__ == "__main__":
    main()
