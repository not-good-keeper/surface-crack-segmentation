"""Rasterise Roboflow COCO-polygon exports into the corpus.

    python dataset/rf_adapt.py --only rf_pipe_crack

Polygons rather than Roboflow PNG masks: they rasterise to the exact boundary at
whatever resolution we pick, with no resampling of an already-resampled label.

Three filters decide what enters -- per-class trust from sources.yaml, category-name
filtering (the pipe export ships `Paper crack` and `Dummy crack` too), and blob
quarantine for coarse anomaly masks. docs/DATASET.md section 4b has the measured gates.

Roboflow augments before export, so the stem before `.rf.` is carried as the group key
and two augmentations of one photo cannot land on opposite sides of a split.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/roboflow"
CFG = Path(__file__).resolve().parent / "sources.yaml"
OUT_IMG = ROOT / "data/clean/images"
OUT_MSK = ROOT / "data/clean/masks"
INDEX = ROOT / "data/clean/index_raw.csv"

# Category name (lowercased) -> our class. Anything unmapped is skipped loudly.
CRACK_WORDS = ("crack", "cracks", "lcd_crack", "pvc pipe crack", "fracture")
SCRATCH_WORDS = ("scratch", "scratches")
# Categories that are structural artefacts of the export, not defects.
IGNORE = {"objects", "object", "phone", "glass", "glass-detection", "molding",
          "null", "background"}
# Out of domain even inside an otherwise-trusted dataset.
OUT_OF_DOMAIN = {"paper crack", "dummy crack", "dirt", "porosity"}

# Measured admission gates, set from the accepted sources' own distributions rather
# than by eye: real thin defects here occupy 0.1-5 % of pixels at 4-15 px mean width.
MAX_FG_FRAC = 0.15
MAX_WIDTH_PX = 20.0


def _skeleton_len(mask):
    from skimage.morphology import skeletonize
    if not mask.any():
        return 0
    return int(skeletonize(mask).sum())


def our_class(cat):
    c = cat.strip().lower()
    if c in IGNORE or c in OUT_OF_DOMAIN:
        return None
    if any(w in c for w in SCRATCH_WORDS):
        return "scratch"
    if any(w in c for w in CRACK_WORDS):
        return "crack"
    return None


def group_key(fname):
    """Roboflow augmentations of one photo share the stem before `.rf.`."""
    stem = Path(fname).stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def adapt(name, spec, rows, size=512):
    d = RAW / name
    jsons = sorted(d.rglob("_annotations.coco.json"))
    if not jsons:
        print(f"[{name}] SKIP (not fetched)")
        return

    blob = spec.get("mask_style") == "blob"
    material = spec.get("material", "unknown")
    allow = {str(u).lower() for u in spec.get("use", [])}
    deny = {str(r).lower() for r in spec.get("reject", [])}

    kept = Counter()
    skipped = Counter()
    for j in jsons:
        ann = json.loads(j.read_text())
        cats = {c["id"]: c["name"] for c in ann.get("categories", [])}
        by_img: dict[int, list] = {}
        for a in ann.get("annotations", []):
            by_img.setdefault(a["image_id"], []).append(a)

        for im in ann.get("images", []):
            ip = j.parent / im["file_name"]
            if not ip.exists():
                continue
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            # One mask per class; the class that owns the image is the one with the
            # most annotated pixels. Mixed-class images are rare here and splitting
            # them would need a multi-class mask the corpus does not carry.
            masks = {"crack": np.zeros((h, w), np.uint8),
                     "scratch": np.zeros((h, w), np.uint8)}
            for a in by_img.get(im["id"], []):
                cname = cats.get(a["category_id"], "")
                cl = our_class(cname)
                if cl is None:
                    skipped[cname] += 1
                    continue
                if deny and cname.strip().lower() in deny:
                    skipped[f"{cname}(rejected)"] += 1
                    continue
                if allow and cname.strip().lower() not in allow and cl not in allow:
                    skipped[f"{cname}(not in use)"] += 1
                    continue
                seg = a.get("segmentation")
                if not seg or not isinstance(seg, list):
                    continue
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(masks[cl], [pts], 255)

            areas = {k: int((v > 0).sum()) for k, v in masks.items()}
            cl = max(areas, key=areas.get)
            if areas[cl] == 0:
                continue

            # Whole-object masks: some annotators outlined the entire phone rather
            # than the crack across it. Real thin defects run 0.1-5 % of pixels.
            frac = areas[cl] / float(h * w)
            if frac > MAX_FG_FRAC:
                skipped[f"whole-object mask ({frac:.0%})"] += 1
                continue

            # Reject compact blobs by shape. A crack or scratch is elongated: its
            # skeleton is long relative to its area. A filled region of the same area
            # has a short skeleton. This catches region-outline annotations that are
            # small enough to pass the fraction test above.
            sk = _skeleton_len(masks[cl] > 0)
            if sk and areas[cl] / max(sk, 1) > MAX_WIDTH_PX:
                skipped[f"blob-shaped (w~{areas[cl]/max(sk,1):.0f}px)"] += 1
                continue

            m = masks[cl]
            if max(h, w) > size:
                s = size / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

            # Blob-style masks are real defects with the wrong mask convention. They
            # enter as hard negatives so the model still sees the imagery, but they
            # never become a pixel target.
            role = "hard_negative" if blob else ("scratch" if cl == "scratch"
                                                 else "positive")
            sid = f"{name}__{group_key(im['file_name'])}__{im['id']}"
            OUT_IMG.mkdir(parents=True, exist_ok=True)
            OUT_MSK.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(OUT_IMG / f"{sid}.png"), img):
                raise IOError(f"failed to write image {sid}")
            if not cv2.imwrite(str(OUT_MSK / f"{sid}.png"),
                               np.zeros_like(m) if blob else m):
                raise IOError(f"failed to write mask {sid}")
            rows.append(dict(name=sid, source=name, sid=sid, material=material,
                             tier="A", role=role,
                             group=f"{name}:{group_key(im['file_name'])}"))
            kept[role] += 1

    note = "  [BLOB -> hard negatives]" if blob else ""
    print(f"[{name}] {dict(kept)} kept{note}"
          + (f"  skipped={dict(skipped)}" if skipped else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")).get("roboflow", {})
    rows: list = []
    for n, s in cfg.items():
        if args.only and n not in args.only:
            continue
        if not s.get("enabled", True):
            print(f"[{n}] DISABLED: {str(s.get('notes','')).strip()[:88]}")
            continue
        adapt(n, s, rows)
    if not rows:
        print("nothing adapted")
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} rows: " + ", ".join(
        f"{k}={v}" for k, v in df.role.value_counts().items()))
    print(df.groupby(["material", "role"]).size().to_string())
    out = ROOT / "data/clean/rf_rows.csv"
    df.to_csv(out, index=False)
    print(f"-> {out}   (merge into {INDEX.name} via dataset/index.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
