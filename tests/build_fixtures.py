"""Assemble the real-image fixture set used by tests/test_model_outputs.py.

    python tests/build_fixtures.py

Small, real, held-out and redistributable -- in that order of importance.

  Real only.      Synthetic composites are what the model trained on. A fixture set
                  built from them would confirm the model reproduces its own training
                  distribution, which is not a test.

  Held out.       Every image comes from a test split, and the builder ASSERTS that no
                  fixture's `group` appears in train or val. The split freeze already
                  guarantees this; asserting it here means a future re-split cannot
                  quietly turn these into training images while the tests keep passing.

  Redistributable. Fixtures live in the repository, so the same licence rule that
                  governs the release archive governs them: openly-licensed sources
                  only, checked against dataset/licences.yaml rather than assumed.

  Balanced.       Defects and clean surfaces, several materials, and a spread of image
                  sizes -- a suite of 400x400 crack close-ups would never exercise the
                  rescaling and geometry paths where the real bugs have been.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
OUT = Path(__file__).resolve().parent / "fixtures"

# (split, role, material, n). Weighted towards test_factory because that is the
# deployment case; negatives included because a model that paints defects on clean
# metal fails in the field regardless of its clDice.
QUOTA = [
    ("test_factory", "positive", "plastic", 6),
    ("test_factory", "positive", "steel", 6),
    ("test_factory", "positive", "ceramic", 4),
    ("test_factory", "positive", "glass", 2),
    ("test_factory_scratch", "scratch", "steel", 6),
    ("test_negatives", "hard_negative", "steel", 4),
    ("test_negatives", "negative", "concrete", 2),
    ("test_negatives", "negative", "wood", 2),
    ("test_seen", "positive", "asphalt", 2),
    ("test_seen", "positive", "plaster", 2),
    ("test_unseen_material", "positive", "wood", 2),
    ("val_unseen_material", "positive", "masonry", 2),
]


def main() -> int:
    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    reg = yaml.safe_load((ROOT / "dataset/licences.yaml").read_text()) or {}
    open_src = {k for k, v in reg.items()
                if isinstance(v, dict) and v.get("redistribute")}

    picked = []
    for split, role, material, n in QUOTA:
        sub = df[(df.split == split) & (df.role == role) & (df.material == material)
                 & df.source.isin(open_src)]
        if not len(sub):
            print(f"  SKIP {split}/{role}/{material}: no openly-licensed rows")
            continue
        # Spread over image size rather than taking the head, so the suite exercises
        # rescaling instead of one convenient resolution.
        sub = sub.assign(_long=sub[["h", "w"]].max(axis=1)).sort_values("_long")
        idx = [int(round(i * (len(sub) - 1) / max(n - 1, 1))) for i in range(min(n, len(sub)))]
        picked.append(sub.iloc[sorted(set(idx))])
    fx = pd.concat(picked).drop_duplicates("name")

    # ---- the assertion that gives the word "unseen" its meaning ------------
    trainval = set(df[df.split.isin(["train", "val", "wood_bg"])].group.astype(str))
    leaked = fx[fx.group.astype(str).isin(trainval)]
    if len(leaked):
        raise SystemExit(
            f"{len(leaked)} fixtures share a parent-image group with train/val:\n"
            f"{leaked[['name', 'split', 'group']].to_string(index=False)}\n"
            f"These are not held-out images and must not be used to test the model.")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "masks").mkdir(parents=True)

    records = []
    for _, r in fx.iterrows():
        img = cv2.imread(str(CLEAN / "images" / f"{r['name']}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        msk = cv2.imread(str(CLEAN / "masks" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(OUT / "images" / f"{r['name']}.png"), img)
        fg = 0
        if msk is not None:
            cv2.imwrite(str(OUT / "masks" / f"{r['name']}.png"), msk)
            fg = int((msk > 127).sum())
        records.append({
            "name": str(r["name"]),
            "split": str(r["split"]),
            "role": str(r["role"]),
            "material": str(r["material"]),
            "source": str(r["source"]),
            "licence": (reg.get(str(r["source"])) or {}).get("licence"),
            "height": int(img.shape[0]),
            "width": int(img.shape[1]),
            "has_defect": bool(fg > 0),
            "gt_defect_px": fg,
            "gt_defect_frac": round(fg / float(img.shape[0] * img.shape[1]), 6),
        })

    n_pos = sum(r["has_defect"] for r in records)
    (OUT / "fixtures.json").write_text(json.dumps({
        "note": "Real, held-out, openly-licensed images for unit-testing the "
                "inspection pipeline. Built by tests/build_fixtures.py; no synthetic "
                "images and nothing that shares a parent-image group with train/val.",
        "split_freeze": "v7",
        "n": len(records),
        "n_with_defect": n_pos,
        "n_clean": len(records) - n_pos,
        "images": records,
    }, indent=1))

    sizes = sorted(max(r["height"], r["width"]) for r in records)
    print(f"{len(records)} fixtures: {n_pos} with defects, {len(records) - n_pos} clean")
    print(f"materials: {sorted({r['material'] for r in records})}")
    print(f"long side: min {sizes[0]}  median {sizes[len(sizes) // 2]}  max {sizes[-1]}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
