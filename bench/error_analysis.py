"""Where the model actually fails, per image rather than in aggregate.

    python bench/error_analysis.py --tag V12_22

A headline clDice is a mean over images whose difficulty varies by an order of
magnitude, and a mean cannot say whether 0.58 means "uniformly mediocre" or "excellent
on most and blind on a few". Those two call for opposite fixes, so this reports the
distribution and what predicts it.

Splits every image's error into the two failure modes that need different responses:

  missed   ground-truth defect pixels the model called background. Recall failure --
           more data, longer training, a loss that weights foreground harder.
  spurious predicted defect pixels with no ground truth nearby. Precision failure --
           harder negatives, a higher threshold, better backgrounds.

Reported against image properties (size, defect fraction, contrast, material) because a
fix has to be aimed at something.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import runs                                      # noqa: E402
from data import CrackPatches                    # noqa: E402
from metrics import evaluate_multiclass          # noqa: E402


def per_image(model, ds, device, bs=16):
    """-> DataFrame, one row per evaluated image."""
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    rows, idx = [], 0
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x.to(device, non_blocking=True))
            P = logits.float().argmax(1).cpu().numpy()
            G = y.numpy()
            for p, g in zip(P, G):
                r = evaluate_multiclass(p[None], g[None])
                r.update(r["any_defect"])
                gt, pr = g > 0, p > 0
                # "Near" a real defect = within 3 px, the tolerance the tolerant-F1
                # metric already uses. Anything further out is not a localisation
                # wobble, it is a different finding.
                near = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                rows.append(dict(
                    i=idx,
                    cldice=float(r.get("cldice", np.nan)),
                    iou=float(r.get("iou", np.nan)),
                    gt_frac=float(gt.mean()),
                    pred_frac=float(pr.mean()),
                    missed_frac=float((gt & ~pr).sum() / max(gt.sum(), 1)),
                    spurious_frac=float((pr & ~near).sum() / max(pr.sum(), 1)),
                ))
                idx += 1
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test_factory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run, ds_kw = runs.load(args.tag, device)
    ds = CrackPatches(args.split, train=False, **ds_kw)
    df = per_image(model, ds, device)

    # Attach the manifest properties, in the dataset's own walk order.
    meta = ds.pos.reset_index(drop=True)
    order = [int(ds._eval_order(len(meta))[i % len(meta)]) for i in range(len(df))]
    df = df.join(meta.iloc[order][["material", "h", "w", "contrast", "blur",
                                   "mean_crack_width"]].reset_index(drop=True))
    df["long_side"] = df[["h", "w"]].max(axis=1)

    print(f"{args.tag}  {args.split}  n={len(df)}  "
          f"prep={ds_kw.get('prep')}  rgb={ds_kw.get('rgb')}\n")

    q = df.cldice.quantile([0, .1, .25, .5, .75, .9, 1.0])
    print("clDice distribution")
    for k, v in q.items():
        print(f"  p{int(k*100):<3} {v:.4f}")
    print(f"  mean {df.cldice.mean():.4f}   "
          f"share below 0.30: {(df.cldice < 0.30).mean():.1%}   "
          f"above 0.70: {(df.cldice > 0.70).mean():.1%}")

    print("\nfailure mode (mean over images)")
    print(f"  missed   {df.missed_frac.mean():.1%} of GT defect pixels called background")
    print(f"  spurious {df.spurious_frac.mean():.1%} of predicted pixels >3 px from any GT")
    print(f"  predicted area / GT area: {df.pred_frac.sum() / max(df.gt_frac.sum(), 1e-9):.2f}x")

    print("\nby material")
    g = df.groupby("material").agg(n=("cldice", "size"), clDice=("cldice", "mean"),
                                   missed=("missed_frac", "mean"),
                                   spurious=("spurious_frac", "mean"))
    print(g.round(3).to_string())

    print("\nwhat predicts a bad image (Spearman vs clDice)")
    for c in ("gt_frac", "long_side", "contrast", "blur", "mean_crack_width"):
        if c in df and df[c].notna().any():
            print(f"  {c:<18}{df.cldice.corr(df[c], method='spearman'):+.3f}")

    worst = df.nsmallest(8, "cldice")
    print("\nworst 8")
    print(worst[["material", "long_side", "gt_frac", "cldice", "missed_frac",
                 "spurious_frac"]].round(4).to_string(index=False))

    out = ROOT / f"data/bench/ERRORS_{args.tag}_{args.split}.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
