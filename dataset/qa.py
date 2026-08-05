"""Verify the frozen evaluation suite before anything trains on it.

Asserts catch structural errors -- leakage, broken pairing, mask polarity, a material
missing from the negative split. `--strict` exits non-zero so they can gate a pipeline.

Contact sheets catch what asserts cannot: a mask that is correctly formatted and points
at the wrong thing. Misalignment and inverted polarity pass every structural test and
are obvious on sight.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
REPORT = ROOT / "data/report"
# Every non-training split. A split missing from this list is not merely unreported --
# it is unchecked: the train<->eval leakage test below intersects against this set, so
# an omitted split reads as clean no matter what leaks into it. Under v4 the scratch
# and steel holdouts were absent here and were never leakage-checked at all.
EVAL_SPLITS = ["val", "val_unseen_material", "test_factory", "test_factory_scratch",
               "test_scratch_blob", "test_seen", "test_unseen_material",
               "test_negatives"]


def sheet(df: pd.DataFrame, title: str, out: Path, n: int = 24, cols: int = 6,
          cell: int = 176):
    if not len(df):
        return
    rows = []
    sub = df.sample(min(n, len(df)), random_state=0)
    cells = []
    for _, r in sub.iterrows():
        img = cv2.imread(str(CLEAN / "images" / f"{r['name']}.png"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(CLEAN / "masks" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if msk is None:
            msk = np.zeros(img.shape[:2], np.uint8)
        img = cv2.resize(img, (cell, cell))
        msk = cv2.resize(msk, (cell, cell), interpolation=cv2.INTER_NEAREST)
        ov = img.copy()
        ov[msk > 127] = (0, 0, 255)
        cells.append(ov)
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    if not rows:
        return
    grid = np.vstack(rows)
    bar = np.zeros((24, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([bar, grid]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--sheets", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    meta = json.loads((ROOT / "data/splits.json").read_text())["meta"]
    errs, notes = [], []

    print(f"splits.json sha256 {meta['sha256'][:16]}...  n={meta['n']}  "
          f"unseen={meta['unseen_material']}")

    # ---- structural asserts -------------------------------------------------
    EVAL = set(EVAL_SPLITS)
    g = df.groupby("group").split.agg(set)
    cross = g[g.map(lambda s: ("train" in s or "wood_bg" in s) and bool(s & EVAL))]
    if len(cross):
        errs.append(f"{len(cross)} groups span train and eval")

    if (df[df.split == "test_unseen_material"].material
            != meta["unseen_material"]).any():
        errs.append("foreign material in test_unseen_material")
    if (df[df.split == "train"].material == meta["unseen_material"]).any():
        errs.append("unseen material leaked into train")

    # The unseen-validation material must be absent from train, val and test_seen or
    # the split is decorative. Without this assert a 1,492-image leak passed as green.
    uv = meta.get("unseen_val_material")
    if uv:
        bad = df[(df.material == uv) & df.split.isin(["train", "val", "test_seen"])]
        if len(bad):
            errs.append(f"unseen-VALIDATION material '{uv}' appears in "
                        f"{sorted(set(bad.split))} ({len(bad)} rows) — early stopping "
                        f"on it would be contaminated")
        if not len(df[df.split == "val_unseen_material"]):
            errs.append(f"unseen_val_material '{uv}' declared but split is empty")

    # Every declared split in splits.json must be non-empty and disjoint by name.
    if df.name.duplicated().any():
        errs.append(f"{int(df.name.duplicated().sum())} duplicate rows in manifest")

    tn = df[df.split == "test_negatives"]
    if (tn.crack_px > 0).any():
        errs.append(f"{int((tn.crack_px > 0).sum())} rows in test_negatives have "
                    f"crack pixels -- they are not negatives")

    # every split must actually be loadable and non-empty
    for s in EVAL_SPLITS:
        if not len(df[df.split == s]):
            errs.append(f"split '{s}' is empty")

    # ...and no split may exist that this file does not know about. Without this, adding
    # a holdout to split.py and forgetting to list it above silently creates a split
    # that no leakage check covers -- which is exactly how the scratch and steel
    # holdouts went unchecked. Fail on the omission, not on its consequences.
    known = set(EVAL_SPLITS) | {"train", "wood_bg"}
    unknown = set(df.split.astype(str)) - known
    if unknown:
        errs.append(f"split(s) {sorted(unknown)} exist in the manifest but are not in "
                    f"EVAL_SPLITS, so nothing above checked them")

    # The three-class head cannot learn a class that never appears in training.
    for role in ("positive", "scratch"):
        n_tr = int(((df.role == role) & (df.split == "train")).sum())
        if not n_tr:
            errs.append(f"no real '{role}' rows in train -- that class would be "
                        f"trained on synthetic data alone")

    # negatives must span the materials we claim to support
    want = {"concrete", "steel", "plastic", "wood", "ceramic", "metal"}
    have = set(tn.material.astype(str))
    if want - have:
        notes.append(f"test_negatives missing materials: {sorted(want - have)}")

    # mask sanity on a sample: strictly binary, pairing intact
    samp = df.sample(min(400, len(df)), random_state=1)
    nonbin = missing = 0
    for _, r in samp.iterrows():
        mp = CLEAN / "masks" / f"{r['name']}.png"
        ip = CLEAN / "images" / f"{r['name']}.png"
        if not mp.exists() or not ip.exists():
            missing += 1
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            missing += 1
            continue
        u = np.unique(m)
        if not set(u.tolist()) <= {0, 255}:
            nonbin += 1
    if missing:
        errs.append(f"{missing}/{len(samp)} sampled rows missing image or mask")
    if nonbin:
        errs.append(f"{nonbin}/{len(samp)} sampled masks are not strictly binary")

    # ---- report -------------------------------------------------------------
    print(f"\n{'split':<22}{'n':>7}{'pos':>7}{'materials'}")
    for s, gg in df.groupby("split"):
        print(f"{s:<22}{len(gg):>7}{int((gg.role=='positive').sum()):>7}  "
              f"{','.join(sorted(set(gg.material.astype(str))))}")

    if args.sheets:
        REPORT.mkdir(parents=True, exist_ok=True)
        for s in EVAL_SPLITS:
            sub = df[df.split == s]
            sheet(sub, f"{s}  (n={len(sub)})", REPORT / f"sheet_{s}.png")
        for m, gg in df[df.split == "test_unseen_material"].groupby("material"):
            sheet(gg, f"unseen:{m}", REPORT / f"sheet_unseen_{m}.png")
        print(f"\ncontact sheets -> {REPORT}")

    for n in notes:
        print(f"\nnote: {n}")
    if errs:
        print("\n*** QA FAILURES ***")
        for e in errs:
            print("  " + e)
        return 1 if args.strict else 0
    print("\nQA PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
