"""Audio-to-video retrieval on held-out VGGSound embeddings (Tab. 5).

One audio-video pair is sampled per class (classes are read from the unique caption
embeddings), audio clips query the videos by cosine similarity, and R@k / MedR are
averaged over 50 sampling seeds.

    python calibration/eval_retrieval.py --embeddings data/vggsound_test_3b.pt \
        --projection calibration/3b/projection.pt
"""
import argparse

import torch
import torch.nn.functional as F


def ranks(q, g):
    s = F.normalize(q, dim=-1) @ F.normalize(g, dim=-1).T
    return (s > s.diagonal().unsqueeze(1)).sum(1).float() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="held-out embeddings with audio / video / text")
    ap.add_argument("--projection", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()

    e = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    a, v = e["audio"].float(), e["video"].float()
    _, cid = torch.unique(e["text"].float(), dim=0, return_inverse=True)
    C = int(cid.max()) + 1
    p = torch.load(args.projection, map_location="cpu", weights_only=False)
    spaces = {
        "Raw": (a, v),
        "Ours": ((a - p["mu_audio"]) @ p["Pa"], (v - p["mu_video"]) @ p["Pv"]),
    }
    res = {k: {"R@1": [], "R@5": [], "R@10": [], "MedR": []} for k in spaces}
    for seed in range(args.seeds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.stack([m[torch.randint(len(m), (1,), generator=g)][0]
                           for m in (torch.nonzero(cid == c, as_tuple=True)[0] for c in range(C))])
        for k, (qa, gv) in spaces.items():
            r = ranks(qa[idx], gv[idx])
            for t in (1, 5, 10):
                res[k][f"R@{t}"].append(100 * (r <= t).float().mean().item())
            res[k]["MedR"].append(r.median().item())
    print(f"classes {C}, seeds {args.seeds}")
    print(f"{'':6}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MedR':>7}")
    for k, m in res.items():
        print(f"{k:6}" + "".join(f"{sum(x) / len(x):7.1f}" for x in m.values()))


if __name__ == "__main__":
    main()
