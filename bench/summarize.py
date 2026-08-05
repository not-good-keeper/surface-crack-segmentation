"""Aggregate benchmark runs into mean +/- std across seeds.

    python bench/summarize.py --tag C3B

Use `summarize3.py` for three-class runs; this one covers every head, including the
binary results that predate the three-class model.

Single-seed numbers are not evidence here: cross-material clDice measured 0.579 / 0.309
/ 0.075 across three seeds of one identical configuration. Runs are matched on the `tag`
inside each JSON, never by globbing filenames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/bench"

SPLITS = ["test_factory", "test_factory_scratch", "test_seen",
          "test_unseen_material", "test_negatives"]
METRICS = ["cldice", "iou", "detect_rate", "fp_area"]


def load(tag_prefix):
    runs = []
    for p in sorted(BENCH.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and str(r.get("tag", "")).startswith(tag_prefix):
            runs.append(r)
    return runs


def agg(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return "     --     "
    if len(v) == 1:
        return f"{v[0]:.4f}       "
    return f"{np.mean(v):.4f} +/-{np.std(v):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="tag prefix, e.g. C3B")
    args = ap.parse_args()

    runs = load(args.tag)
    if not runs:
        print(f"no runs with tag prefix {args.tag!r} in {BENCH}")
        return 1

    seeds = [r["args"]["seed"] for r in runs]
    print(f"{len(runs)} run(s), tag {args.tag}*, seeds {seeds}")
    print(f"model    {runs[0]['model']}")
    c = runs[0].get("cost", {})
    print(f"cost     {c.get('params', 0)/1e6:.2f} M params, "
          f"{c.get('cpu_ms', float('nan')):.0f} ms CPU (desktop, single process)")
    if len(runs) < 3:
        print("WARNING: fewer than 3 seeds -- the spread below is not a variance "
              "estimate, and this project has seen 7x swings between seeds.")

    print(f"\n{'split':<24}" + "".join(f"{m:>18}" for m in METRICS))
    for s in SPLITS:
        row = f"{s:<24}"
        for m in METRICS:
            row += f"{agg([r.get(s, {}).get(m) for r in runs]):>18}"
        print(row)

    print(f"\n{'class recall (defect px)':<24}{'crack':>18}{'scratch':>18}")
    for s in ("test_factory", "test_factory_scratch", "test_scratch_blob"):
        cr = [r.get(s, {}).get("crack_class_recall") for r in runs]
        sc = [r.get(s, {}).get("scratch_class_recall") for r in runs]
        print(f"{s:<24}{agg(cr):>18}{agg(sc):>18}")

    base = runs[0].get("baseline_all_background")
    if base:
        # The floor for the whole metric set. Printed with the results, not filed away,
        # so nobody has to take on trust that the numbers above mean anything.
        print(f"\nall-background baseline on test_factory: "
              f"pixel_accuracy={base.get('pixel_accuracy', float('nan')):.4f} "
              f"but clDice={base.get('cldice', 0):.4f}, "
              f"detect_rate={base.get('detect_rate', 0):.4f}")
        print("  -- a model predicting nothing scores that accuracy. This is why "
              "accuracy is never the headline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
