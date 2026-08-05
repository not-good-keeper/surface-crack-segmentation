"""Verified crack-free background pool, grouped by material.

This module exists because of a single failure mode that is easy to create and
almost invisible afterwards: compositing a synthetic crack onto a background that
already contains a real, unlabelled crack. The resulting sample teaches the model
that real cracks are background -- the exact opposite of the goal -- and no amount
of downstream QA on mask alignment will reveal it, because the synthetic mask is
perfectly correct.

So a background is admitted only if ALL of these hold:
  1. role == 'negative' in the manifest (never 'positive', never 'hard_negative' --
     hard negatives are real defects and may include genuine cracks)
  2. its own mask is completely empty
  3. its source is on the allow-list of pools whose cracked half was excluded at
     adapter time by folder or label, not by inference

These are construction-level guarantees, which is why they are the whole defence.

A fourth rule was attempted and DELIBERATELY REMOVED: an unsupervised ridge screen
(black-hat and Frangi vesselness) meant to catch unlabelled dark linear structure.
Measured on 50 known-cracked vs 50 known-clean images it does not discriminate --
Frangi scored 7.87 vs 7.51 (medians), black-hat 0.004 vs 0.001 with p25/p75 ranges
that overlap almost completely. The reason is the same reason this project is hard:
industrial surfaces are full of thin dark linear structures (wood grain, brushed
metal, form lines, tar seams) that are indistinguishable from cracks to any
low-level filter. Shipping it with an arbitrary threshold would have rejected
either nothing or a random subset while implying a safety guarantee it cannot make.
The honest substitutes are rules 1-3 plus the per-material visual contact sheets
in qa.py, and -- once a model exists -- re-screening the pool with the model itself.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
OUT = ROOT / "data/backgrounds"
BG_SIZE = 512   # render above target 256 so downsampling anti-aliases

# Pools we have verified are crack-free by construction (see dataset/adapters.py):
#   crackseg9k  -> only the 'noncrack_*' subset is role=negative
#   sdnet       -> only Non-cracked/, the Cracked/ half is excluded entirely
#   ozgenel     -> only the Negative/ folder
#   dtd         -> the 'cracked' category is excluded
#   kolektor    -> only images whose GT mask is empty
#   magnetic    -> only MT_Free
#   severstal   -> only images absent from train.csv defect list
ALLOWED_SOURCES = {"crackseg9k", "sdnet", "ozgenel", "dtd",
                   "kolektor", "magnetic_tile", "severstal", "gc10", "wood"}

# gc10/neu_det/severstal-defective are hard negatives: real metal defects. They are
# excellent training negatives but must NOT be synthesis substrates, because a
# synthetic crack drawn over a real defect produces an ambiguous label.
SUBSTRATE_EXCLUDE_ROLES = {"positive", "hard_negative"}


def usable_substrate(gray):
    """Reject images that cannot carry a visible crack.

    A synthetic crack darkens what is under it, so a near-black region has no
    dynamic range left to darken and the crack renders invisible -- but the mask
    still claims it, which trains the model to hallucinate. Blown-out white areas
    fail the same way, and largely-flat frames (blurred backdrops, strip edges)
    give the model no texture to distinguish crack from substrate.
    """
    m, s = float(gray.mean()), float(gray.std())
    if not (28.0 <= m <= 228.0):
        return False                        # too dark or blown out overall
    if s < 9.0:
        return False                        # featureless
    dark = float((gray < 18).mean())
    blown = float((gray > 242).mean())
    if dark > 0.35 or blown > 0.35:
        return False                        # large crushed/clipped regions
    return True


def ridge_score(gray):
    """Rough 'does this contain a dark linear structure' score.

    Black-hat isolates thin dark features; we then ask how much of the image is
    covered by strong, elongated dark ridges. Real cracks score high; texture,
    grain and noise score low because they are not sustained ridges.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    if bh.max() < 8:
        return 0.0
    thr = max(12, int(np.percentile(bh, 99.5)))
    ridge = (bh >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ridge, 8)
    score = 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        elong = max(w, h) / max(1.0, min(w, h))
        if elong >= 3.0:                     # long and thin => crack-like
            score += area * min(elong, 12.0)
    return float(score / gray.size)


def _restrict_to_train_side(df: pd.DataFrame) -> pd.DataFrame:
    """Drop anything held out for evaluation.

    Synthetic samples become TRAINING data, so building them on val/test images would
    put evaluation pixels into the training set through the back door -- invisible to
    a split check, because the synthetic patch has a new id. Only `train` and
    `wood_bg` (the designated synthesis substrate) are eligible.
    """
    sp = ROOT / "data/manifest_split.csv"
    if not sp.exists():
        print("[bg] WARNING: no manifest_split.csv yet -- cannot exclude eval images. "
              "Re-run this after dataset/split.py.")
        return df
    s = pd.read_csv(sp)[["name", "split"]]
    before = len(df)
    df = df.merge(s, on="name", how="left")
    df = df[df.split.isin(["train", "wood_bg"])]
    print(f"[bg] split filter: {before} -> {len(df)} (train + wood_bg only)")
    return df


def build(limit_per_material, ridge_thresh, workers=1):
    df = pd.read_csv(ROOT / "data/manifest.csv")
    df = df[(df.role == "negative") & (~df.role.isin(SUBSTRATE_EXCLUDE_ROLES))]
    df = _restrict_to_train_side(df)
    df = df[df.source.isin(ALLOWED_SOURCES)]
    df = df[df.crack_px == 0]           # rule 2: mask must be empty
    print(f"[bg] {len(df)} candidates after role/source/mask filters")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], 0
    for mat, g in df.groupby("material"):
        kept = 0
        (OUT / mat).mkdir(exist_ok=True)
        for _, r in g.iterrows():
            if limit_per_material and kept >= limit_per_material:
                break
            p = CLEAN / "images" / f"{r['name']}.png"
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < 96:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s = ridge_score(gray)           # recorded for later analysis, NOT a filter
            if not usable_substrate(gray):
                rejected += 1
                continue
            img = cv2.resize(img, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_AREA)
            out = OUT / mat / f"{r['name']}.png"
            cv2.imwrite(str(out), img)
            rows.append(dict(bg_id=r["name"], material=mat, source=r["source"],
                             ridge=s, path=str(out.relative_to(ROOT))))
            kept += 1
        print(f"  {mat:<10} kept {kept}")

    idx = pd.DataFrame(rows)
    idx.to_csv(ROOT / "data/backgrounds.csv", index=False)
    print(f"\n[bg] {len(idx)} backgrounds, {rejected} rejected by ridge screen")
    print(f"[bg] by material:\n{idx.material.value_counts().to_string()}")
    return idx


def harvest_from_positives(limit_per_material, min_side=160,
                           tries: int = 12):
    """Cut crack-FREE crops out of crack images.

    Some materials (asphalt, masonry) appear in this corpus only as crack photos,
    so a role-based filter leaves them with zero synthesis substrates. But a region
    of a crack photo where the mask is empty is exactly what we need: the right
    material, the right camera, the right lighting, and verified crack-free by the
    same annotation we already trust for the positives.

    A crop is accepted only if the mask is empty over the crop AND over a margin
    around it, so a crack just outside the window cannot bleed in after resizing.
    """
    df = pd.read_csv(ROOT / "data/manifest.csv")
    pos = _restrict_to_train_side(df[(df.role == "positive")])
    rng = np.random.default_rng(0)
    rows = []
    for mat, g in pos.groupby("material"):
        kept = 0
        (OUT / mat).mkdir(parents=True, exist_ok=True)
        for _, r in g.sample(frac=1.0, random_state=1).iterrows():
            if kept >= limit_per_material:
                break
            img = cv2.imread(str(CLEAN / "images" / f"{r['name']}.png"), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(CLEAN / "masks" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            h, w = msk.shape
            side = min(h, w, max(min_side, int(min(h, w) * 0.6)))
            if side < min_side:
                continue
            for _ in range(tries):
                y = int(rng.integers(0, max(1, h - side)))
                x = int(rng.integers(0, max(1, w - side)))
                m = max(8, side // 12)              # safety margin
                y0, y1 = max(0, y - m), min(h, y + side + m)
                x0, x1 = max(0, x - m), min(w, x + side + m)
                if msk[y0:y1, x0:x1].any():
                    continue                        # crack in or near the window
                crop = img[y:y + side, x:x + side]
                if crop.size == 0 or not usable_substrate(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
                    continue
                crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_AREA)
                bid = f"{r['name']}_crop{kept}"
                p = OUT / mat / f"{bid}.png"
                cv2.imwrite(str(p), crop)
                rows.append(dict(bg_id=bid, material=mat, source=f"{r['source']}_crop",
                                 ridge=-1.0, path=str(p.relative_to(ROOT))))
                kept += 1
                break
        print(f"  {mat:<10} harvested {kept} crack-free crops")
    return pd.DataFrame(rows)


def harvest_crops_from_split(split, material, per_image=12,
                             min_side: int = 200):
    """Multiply substrate variety for a material that has very few whole images.

    Wood is the case this exists for: nearly every board in the corpus contains a
    crack, so only 68 boards are clean AND non-leaking, and after the quality filter
    just 24 survive as whole-image substrates. Cutting several crops from each of
    those 68 boards raises substrate count without touching a single test board --
    the board-level partition in dataset/split.py still holds.
    """
    sp = ROOT / "data/manifest_split.csv"
    if not sp.exists():
        return pd.DataFrame()
    df = pd.read_csv(sp)
    df = df[(df.split == split) & (df.material == material)]
    rng = np.random.default_rng(7)
    rows = []
    (OUT / material).mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        img = cv2.imread(str(CLEAN / "images" / f"{r['name']}.png"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(CLEAN / "masks" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        if side < min_side:
            continue
        kept = 0
        for _ in range(per_image * 4):
            if kept >= per_image:
                break
            s = int(rng.integers(min_side, side + 1))
            y = int(rng.integers(0, h - s + 1))
            x = int(rng.integers(0, w - s + 1))
            if msk is not None and msk[y:y + s, x:x + s].any():
                continue
            crop = img[y:y + s, x:x + s]
            if not usable_substrate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
                continue
            crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_AREA)
            bid = f"{r['name']}_c{kept}"
            p = OUT / material / f"{bid}.png"
            cv2.imwrite(str(p), crop)
            rows.append(dict(bg_id=bid, material=material,
                             source=f"{r['source']}_bgcrop", ridge=-1.0,
                             path=str(p.relative_to(ROOT))))
            kept += 1
    print(f"  {material:<10} {len(rows)} crops from {len(df)} '{split}' images")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wood-bg-crops", action="store_true",
                    help="multiply wood substrates by cropping the wood_bg split")
    ap.add_argument("--from-positives", action="store_true",
                    help="also harvest crack-free crops out of crack images")
    ap.add_argument("--limit-per-material", type=int, default=2500)
    ap.add_argument("--ridge-thresh", type=float, default=0.85,
                    help="lower = stricter; tune with --calibrate first")
    ap.add_argument("--calibrate", action="store_true",
                    help="print ridge scores for known-crack vs known-clean images")
    args = ap.parse_args()

    if args.calibrate:
        df = pd.read_csv(ROOT / "data/manifest.csv")
        for role, n in (("positive", 60), ("negative", 60)):
            sub = df[df.role == role].sample(min(n, (df.role == role).sum()),
                                             random_state=0)
            sc = []
            for _, r in sub.iterrows():
                img = cv2.imread(str(CLEAN / "images" / f"{r['name']}.png"),
                                 cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    sc.append(ridge_score(img))
            sc = np.array(sc)
            print(f"{role:<10} n={len(sc):<4} median={np.median(sc):8.3f} "
                  f"p25={np.percentile(sc,25):8.3f} p75={np.percentile(sc,75):8.3f} "
                  f"p95={np.percentile(sc,95):8.3f}")
        print("\nPick --ridge-thresh between the negative p75 and the positive median.")
        return 0

    idx = build(args.limit_per_material, args.ridge_thresh)
    if args.wood_bg_crops:
        print("\n[bg] harvesting extra substrates from wood_bg ...")
        idx = pd.concat([idx, harvest_crops_from_split("wood_bg", "wood")],
                        ignore_index=True)
    if args.from_positives:
        print("\n[bg] harvesting crack-free crops from positive images ...")
        extra = harvest_from_positives(args.limit_per_material)
        idx = pd.concat([idx, extra], ignore_index=True)
        idx.to_csv(ROOT / "data/backgrounds.csv", index=False)
        print(f"\n[bg] TOTAL {len(idx)} backgrounds")
        print(idx.material.value_counts().to_string())

if __name__ == "__main__":
    main()
