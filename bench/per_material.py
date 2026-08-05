"""Break the `test_factory` headline down by material (ADR-016).

    python bench/per_material.py --tag V8A_11 --by source

`test_factory` is not one population -- epoxy has no training data at all, plastic has
749 images of one PVC pipe product. A headline that rises because plastic arrived is not
the same result as one that rises because unseen surfaces got easier, and the mean
cannot tell those apart.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs                                            # noqa: E402
from data import CrackPatches                         # noqa: E402
from metrics import evaluate_multiclass               # noqa: E402
from train import predict_split                       # noqa: E402

SPLIT = "test_factory"          # the headline; breaking down anything else is a different question
MIN_TRUSTWORTHY = 25


def subset(ds, mask) -> bool:
    """Restrict an eval dataset to the rows `mask` selects from ds.pos.

    Evaluation indexes ds.pos directly, so swapping the frame and recomputing the length
    is enough -- no loader surgery, and nothing to change in data.py.
    """
    ds.pos = ds.pos[mask].reset_index(drop=True)
    if not len(ds.pos):
        return False
    ds.length = ds._eval_len()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run tag, e.g. V8A_11")
    ap.add_argument("--by", default="material", choices=["material", "source"])
    ap.add_argument("--eval-batches", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run, ds_kw = runs.load(args.tag, device)
    classes = ds_kw["classes"]
    probe = CrackPatches(SPLIT, train=False, **ds_kw)
    groups = sorted(probe.pos[args.by].unique())
    print(f"{args.tag}  {SPLIT}  by {args.by}  "
          f"(best_epoch={run.get('best_epoch', 'final')})\n")
    print(f"{args.by:<14}{'n':>5}{'IoU':>9}{'clDice':>9}{'det':>8}"
          f"{'crack rec':>11}{'scr rec':>9}")

    rows = []
    for g in groups:
        ds = CrackPatches(SPLIT, train=False, **ds_kw)
        if not subset(ds, ds.pos[args.by].to_numpy() == g):
            continue
        n = len(ds.pos)
        P, G = predict_split(model, ds, device, max_batches=args.eval_batches,
                             classes=classes)
        r = evaluate_multiclass(P, G) if classes > 1 else None
        if r is None:
            continue
        r.update({k: v for k, v in r["any_defect"].items()})
        flag = "  <- n too small to quote" if n < MIN_TRUSTWORTHY else ""
        print(f"{str(g):<14}{n:>5}{r['iou']:>9.4f}{r['cldice']:>9.4f}"
              f"{r['detect_rate']:>8.3f}{r['crack_class_recall']:>11.3f}"
              f"{r['scratch_class_recall']:>9.3f}{flag}")
        rows.append((g, n, r))

    if rows:
        # The unweighted mean over materials is NOT the headline: the headline is
        # weighted by how many images each material happens to contribute, so a large
        # easy material can carry it. Printing both makes that visible.
        w = np.array([n for _, n, _ in rows], float)
        cl = np.array([r["cldice"] for _, _, r in rows], float)
        print(f"\nimage-weighted clDice (= the headline): "
              f"{float((w * cl).sum() / w.sum()):.4f}")
        print(f"material-weighted clDice (each material equal): {float(cl.mean()):.4f}")
        print("A gap between these two means the headline is being carried by "
              "whichever material contributes the most images.")

if __name__ == "__main__":
    main()
