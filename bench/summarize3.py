"""Aggregate three-class runs into a table with seed variance.

The three-class counterpart to `summarize.py`: same evidence, but it prints every seed
rather than only the spread, because a std over three points hides which seed was the
outlier.

    python bench/summarize3.py --tag C3F

Reports mean +/- std across seeds, never a single run. Cross-material clDice varied by
+/-0.055 between seeds of an identical configuration in this project, which is large
enough that a single-seed number can support the opposite conclusion from the truth.
Any row backed by fewer than three seeds is marked, because a std over two points is
not a spread.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/bench"

# (split, metric, label) -- the splits the architecture actually makes claims about.
VIEWS = [
    ("test_factory",         "cldice",      "factory crack clDice"),
    ("test_factory",         "iou",         "factory crack IoU"),
    ("test_factory",         "detect_rate", "factory crack detection"),
    ("test_factory",         "crack_class_recall", "crack class recall"),
    ("test_factory_scratch", "cldice",      "factory scratch clDice"),
    ("test_factory_scratch", "iou",         "factory scratch IoU"),
    ("test_factory_scratch", "detect_rate", "factory scratch detection"),
    ("test_factory_scratch", "scratch_class_recall", "scratch class recall"),
    ("test_seen",            "cldice",      "civil (auxiliary) clDice"),
    ("test_unseen_material", "cldice",      "wood transfer clDice"),
    ("test_unseen_material", "detect_rate", "wood transfer detection"),
    ("test_negatives",       "fp_area",     "FP area on clean surfaces"),
]

TARGETS = {
    "factory crack detection": (0.75, "ge"),
    "factory scratch detection": (0.75, "ge"),
    "FP area on clean surfaces": (0.005, "le"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="C3F", help="substring of the run tag")
    args = ap.parse_args()

    runs = []
    for p in sorted(BENCH.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if args.tag in str(d.get("tag", "")):
            runs.append((p.stem, d))
    if not runs:
        print(f"no runs matching tag {args.tag!r} in {BENCH}")
        return 1

    print(f"{len(runs)} run(s): {', '.join(n for n, _ in runs)}")
    cost = runs[0][1].get("cost", {})
    print(f"model {runs[0][1]['model']} | {cost.get('params', 0)/1e6:.2f} M params | "
          f"{cost.get('cpu_ms', float('nan')):.0f} ms CPU (desktop, single run)\n")

    print(f"{'metric':<28}{'mean':>9}{'std':>9}{'n':>4}   {'per seed'}")
    print("-" * 78)
    for split, metric, label in VIEWS:
        vals = []
        for _, d in runs:
            r = d.get(split)
            if isinstance(r, dict) and metric in r:
                v = r[metric]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
        if not vals:
            continue
        m, s = float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)
        flag = ""
        if label in TARGETS:
            tgt, dirn = TARGETS[label]
            ok = (m >= tgt) if dirn == "ge" else (m <= tgt)
            flag = f"  {'MEETS' if ok else 'MISSES'} target {tgt}"
        warn = "" if len(vals) >= 3 else "  (<3 seeds: std not meaningful)"
        print(f"{label:<28}{m:>9.4f}{s:>9.4f}{len(vals):>4}   "
              f"{', '.join(f'{v:.3f}' for v in vals)}{flag}{warn}")

    # The floor. If this is not printed alongside the numbers above, a reader has no
    # way to know that predicting nothing scores 96-98 % pixel accuracy on this data.
    base = runs[0][1].get("baseline_all_background")
    if base:
        print(f"\nall-background baseline on test_factory: "
              f"pixel_accuracy={base.get('pixel_accuracy', float('nan')):.4f}  "
              f"clDice={base.get('cldice', 0):.4f}  "
              f"detect_rate={base.get('detect_rate', 0):.4f}")
        print("  -- a model that predicts nothing scores that pixel accuracy. It is "
              "why clDice and detection lead this table and accuracy is absent.")

    blob = runs[0][1].get("test_scratch_blob")
    if blob and "note" in blob:
        rec = [d.get("test_scratch_blob", {}).get("scratch_class_recall")
               for _, d in runs]
        rec = [r for r in rec if r is not None and not np.isnan(r)]
        print(f"\ntest_scratch_blob (MVTec, detection/class only): "
              f"detect={blob.get('detect_rate', float('nan')):.3f}"
              + (f", scratch class recall {np.mean(rec):.3f}" if rec else ""))
        print(f"  -- {blob['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
