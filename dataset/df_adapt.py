"""Project DefectForge's 16-class store onto our format.

The crack family (crack, hairline, fatigue, branch, spider, micro, long, fracture)
becomes foreground; scratch, corrosion, rust, paint_peel, chip, pit and dent become
background hard negatives. Same engine and optics on both sides, so they can only be
told apart by structure.

Mixed samples keep the crack pixels and drop the rest to background. Output matches
defectforge/generate.py so bench/data.py loads it unchanged.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIZE = 256

CRACK_CLASSES = {
    "crack", "hairline_crack", "fatigue_crack", "branch_crack",
    "spider_crack", "micro_crack", "long_crack", "fracture",
}
# Scratches get their own mask and their own kind. For the binary crack task they
# are hard negatives; for the two-class task they are the second foreground class.
SCRATCH_CLASSES = {"scratch", "deep_scratch"}


def _one(args):
    sample_dir, out_dir = args
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    try:
        meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None

    img = cv2.imread(str(sample_dir / "image.png"), cv2.IMREAD_COLOR)
    if img is None:
        return None

    # semantic_mask stores class_id + 1, 0 = background
    sem = cv2.imread(str(sample_dir / "semantic_mask.png"), cv2.IMREAD_UNCHANGED)
    classes = meta.get("defect_types") or []

    mask = np.zeros(img.shape[:2], np.uint8)
    smask = np.zeros(img.shape[:2], np.uint8)
    has_crack = has_scratch = False
    if sem is not None and sem.ndim == 2:
        # Embedded rather than imported: the import fails silently in spawned
        # workers on Windows and every sample falls through to the crack-only
        # fallback, which looks like a successful run.
        try:
            for cid in np.unique(sem):
                if cid == 0:
                    continue
                name = DF_CLASSES[int(cid) - 1] if int(cid) - 1 < len(DF_CLASSES) else ""
                if name in CRACK_CLASSES:
                    mask[sem == cid] = 255
                    has_crack = True
                elif name in SCRATCH_CLASSES:
                    smask[sem == cid] = 255
                    has_scratch = True
        except Exception:  # noqa: BLE001
            # fall back to the binary mask when the taxonomy import is unavailable,
            # but only if every defect in the sample is a crack -- otherwise we would
            # label a scratch as a crack
            if classes and all(c in CRACK_CLASSES for c in classes):
                bm = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
                if bm is not None:
                    mask = ((bm > 127) * 255).astype(np.uint8)
                    has_crack = bool(mask.any())

    if img.shape[0] != SIZE or img.shape[1] != SIZE:
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        smask = cv2.resize(smask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    # A crack takes precedence: if a sample has both, it is a crack sample and the
    # scratch pixels stay background -- same photo, both structures, strongest
    # possible discrimination signal for the binary task.
    if has_crack:
        kind = "crack"
    elif has_scratch:
        kind, mask = "scratch", smask
    else:
        kind = "negative"
    pid = f"df_{kind}_{meta.get('uuid', sample_dir.name)[:16]}"
    if not cv2.imwrite(str(out_dir / "images" / f"{pid}.png"), img):
        return None
    cv2.imwrite(str(out_dir / "masks" / f"{pid}.png"), mask)

    mat = meta.get("material", "unknown")
    return dict(patch_id=pid, kind=kind, material=mat,
                bg_id=f"dfsynth:{mat}", bg_path="",
                rng_seed=meta.get("seed", -1), sample_index=-1,
                generator_version="defectforge-engine-1.0",
                crack_px=int((mask > 0).sum()),
                crack_px_frac=float((mask > 0).mean()),
                meta=json.dumps({"classes": classes,
                                 "severity": meta.get("severity")}, default=float))


# defectforge.taxonomy.CLASSES, in order. semantic_mask stores class_id + 1.
DF_CLASSES = ("crack", "hairline_crack", "fatigue_crack", "branch_crack",
              "spider_crack", "micro_crack", "long_crack", "fracture",
              "scratch", "deep_scratch", "corrosion", "rust", "paint_peel",
              "chip", "pit", "dent")

# Location of the DefectForge source checkout; override with --df-root.
DF_ROOT = ROOT / "data" / "DefectForge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/df_synth")
    ap.add_argument("--out", default="data/df_patches")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--df-root", default=str(DF_ROOT))
    args = ap.parse_args()

    globals()["DF_ROOT"] = Path(args.df_root)
    src, out = ROOT / args.src, ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    dirs = [p.parent for p in src.rglob("image.png")]
    print(f"[df_adapt] {len(dirs)} samples -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_one, [(str(d), str(out)) for d in dirs],
                                     chunksize=32), 1):
            if r:
                rows.append(r)
            if i % 5000 == 0:
                print(f"  ... {i}/{len(dirs)} kept={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    prov = out.parent / f"{out.name}_provenance.parquet"
    df.to_parquet(prov, index=False)
    n_c = int((df.kind == "crack").sum()) if len(df) else 0
    print(f"\n[df_adapt] {len(df)} patches: {n_c} crack / {len(df)-n_c} hard-negative")
    if len(df):
        print(df.groupby(["material", "kind"]).size().to_string())
        print(f"[df_adapt] mean fg on cracks: "
              f"{100*df[df.kind=='crack'].crack_px_frac.mean():.2f}%")
    print(f"[df_adapt] provenance -> {prov}")

if __name__ == "__main__":
    main()
