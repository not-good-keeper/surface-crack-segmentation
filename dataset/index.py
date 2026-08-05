"""Build data/manifest.csv -- per-image statistics that drive every later decision.

The column that matters most is mean_crack_width. Public crack datasets disagree
about what a mask means: some annotate the full visible crack width (~8-15 px),
others annotate a 1-2 px skeleton down the centreline. Mixed together they hand a
small model two contradictory targets for identical visual evidence, which is the
single largest source of gradient noise available to us. Measuring the width
distribution per source is how we find those sources and drop them.

    mean_crack_width = 2 * mean( distanceTransform(mask)[skeleton] )

because the distance transform at a skeleton pixel is the local half-width.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
OUT = ROOT / "data/manifest.csv"

try:
    import imagehash
except ImportError:
    imagehash = None


def stats_one(name: str) -> dict | None:
    ip = CLEAN / "images" / f"{name}.png"
    mp = CLEAN / "masks" / f"{name}.png"
    img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
    if img is None:
        return None
    msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        msk = np.zeros(img.shape[:2], np.uint8)

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    binm = (msk > 127).astype(np.uint8)
    crack_px = int(binm.sum())

    mean_width = skel_len = n_comp = 0.0
    if crack_px:
        dist = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(binm.astype(bool))
        except Exception:  # noqa: BLE001
            sk = binm.astype(bool)
        skel_len = float(sk.sum())
        if skel_len:
            mean_width = float(2.0 * dist[sk].mean())
        n_comp = float(cv2.connectedComponents(binm)[0] - 1)

    ph = ""
    if imagehash is not None:
        try:
            ph = str(imagehash.phash(Image.fromarray(
                cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB))))
        except Exception:  # noqa: BLE001
            ph = ""

    return dict(
        name=name, h=h, w=w,
        crack_px=crack_px, crack_px_frac=crack_px / (h * w),
        mean_crack_width=mean_width, skel_len=skel_len, n_components=n_comp,
        blur=float(cv2.Laplacian(g, cv2.CV_64F).var()),
        brightness=float(g.mean()), contrast=float(g.std()),
        phash=ph,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    idx = pd.read_csv(CLEAN / "index_raw.csv")
    names = idx["name"].tolist()
    print(f"[index] {len(names)} images, {args.workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(stats_one, names, chunksize=64), 1):
            if r:
                rows.append(r)
            if i % 2000 == 0:
                print(f"  ... {i}/{len(names)}", flush=True)

    df = pd.DataFrame(rows).merge(idx, on="name", how="left", suffixes=("", "_raw"))
    df.to_csv(OUT, index=False)
    print(f"[index] wrote {OUT}  ({len(df)} rows)")

    # ---- the audit that decides which sources survive -----------------------
    pos = df[(df.role == "positive") & (df.crack_px > 0)]
    print("\n=== crack-width audit (positives only) ===")
    print(f"{'source':<16}{'n':>6}{'mean_w':>9}{'median_w':>10}{'p10':>7}{'p90':>7}"
          f"{'fg%':>7}  verdict")
    for src, g in pos.groupby("source"):
        mw, med = g.mean_crack_width.mean(), g.mean_crack_width.median()
        p10, p90 = g.mean_crack_width.quantile([.1, .9])
        fg = g.crack_px_frac.mean() * 100
        verdict = ("SKELETON -> drop" if med < 3.0 else
                   "thin" if med < 5.0 else "full-width OK")
        print(f"{src:<16}{len(g):>6}{mw:>9.2f}{med:>10.2f}{p10:>7.2f}{p90:>7.2f}"
              f"{fg:>7.2f}  {verdict}")

    print("\n=== by material ===")
    for mat, g in df.groupby("material"):
        print(f"  {mat:<12} {len(g):>6} imgs | "
              f"{(g.role=='positive').sum():>5} pos | "
              f"{(g.role=='hard_negative').sum():>5} hard-neg | "
              f"{(g.role=='negative').sum():>5} neg")

    if "phash" in df and df.phash.notna().any():
        dup = df[df.phash != ""].duplicated("phash", keep=False).sum()
        print(f"\n[index] {dup} images share a phash with at least one other "
              f"(exact near-duplicates; normalize.py removes them)")

if __name__ == "__main__":
    main()
