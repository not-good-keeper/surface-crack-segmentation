"""Definitive evaluation of one checkpoint: every frozen split, no subsampling.

    python bench/final_eval.py --tag V8A_22

Training caps each split at `--eval-batches 40` = 1,280 images. Fine during a sweep, but
`test_negatives` holds 4,626 clean images, so every false-positive figure -- including
the NFR-03 pass -- would rest on a quarter of them.

It also reports both decision rules from the same logits. `bench/train.py` uses plain
argmax; the app applies ARCHITECTURE.md 7.1, so training-time metrics did not describe
the deployed model. Thresholds come from `app.postprocess.Profile` rather than being
restated here, and were selected on `val`, never on a split reported below.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import runs                                                  # noqa: E402
from data import CrackPatches                               # noqa: E402
from metrics import evaluate_multiclass, all_background_baseline  # noqa: E402
from app.postprocess import Profile, class_map               # noqa: E402

BENCH = ROOT / "data/bench"
SPLITS = ["test_factory", "test_factory_scratch", "test_scratch_blob", "test_seen",
          "test_unseen_material", "test_negatives", "val_unseen_material"]
# Geometry against these is agreement with a mask convention we rejected on a width
# audit (33.6 px median vs 4-15 px for accepted thin structures), so only detection
# and class survive.
DETECTION_ONLY = {"test_scratch_blob"}


def predictions(model, ds, device, profile, bs=32):
    """Both decision rules plus ground truth, as uint8 class maps.

    Not probabilities: holding (4626,3,256,256) float32 beside int64 targets wants ~6 GB
    and fails outright. Applying both rules per batch and keeping only the class maps
    costs ~300 MB for identical results.
    """
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    A, S, G = [], [], []
    with torch.no_grad():
        for x, y in dl:                       # NO cap: this is the point of the script
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x.to(device, non_blocking=True))
            p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            A.append(p.argmax(axis=1).astype(np.uint8))
            S.append(class_map(p, profile).astype(np.uint8))
            G.append(y.numpy().astype(np.uint8))
    return np.concatenate(A), np.concatenate(S), np.concatenate(G)


def flat(pred, G):
    r = evaluate_multiclass(pred, G)
    r.update(r["any_defect"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    prof = Profile()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run, ds_kw = runs.load(args.tag, device)

    print(f"{args.tag}  FULL evaluation, no subsampling")
    print(f"profile '{prof.profile_id}': crack>{prof.crack_thresh} "
          f"scratch>{prof.scratch_thresh}   (argmax shown for comparison)\n")
    print(f"{'split':<22}{'n':>6}{'rule':>8}{'IoU':>9}{'clDice':>9}{'det':>7}"
          f"{'fp_area':>10}{'crackR':>8}{'scrR':>7}")

    out = {"tag": args.tag, "profile_id": prof.profile_id,
           "crack_thresh": prof.crack_thresh, "scratch_thresh": prof.scratch_thresh,
           "checkpoint": run["checkpoint"], "splits": {}}
    for sp in SPLITS:
        ds = CrackPatches(sp, train=False, **ds_kw)
        n = len(ds)
        if not n:
            continue
        t0 = time.time()
        A, S, G = predictions(model, ds, device, prof, bs=args.batch)
        rows = {"argmax": flat(A, G), "7.1": flat(S, G)}
        def cell(value, fmt=".4f", show=True):
            if not show or value is None or (isinstance(value, float)
                                             and np.isnan(value)):
                return "--"
            return format(value, fmt)

        for rule, r in rows.items():
            geo = sp not in DETECTION_ONLY
            first = rule == "argmax"
            print(f"{sp if first else '':<22}{n if first else '':>6}{rule:>8}"
                  f"{cell(r.get('iou'), show=geo):>9}"
                  f"{cell(r.get('cldice'), show=geo):>9}"
                  f"{cell(r.get('detect_rate'), '.3f'):>7}"
                  f"{cell(r.get('fp_area'), '.5f'):>10}"
                  f"{cell(r.get('crack_class_recall'), '.3f'):>8}"
                  f"{cell(r.get('scratch_class_recall'), '.3f'):>7}")
        if sp == "test_factory":
            out["baseline_all_background"] = all_background_baseline(G > 0)
        out["splits"][sp] = {"n": n, "seconds": round(time.time() - t0, 1),
                             **{k: {kk: vv for kk, vv in v.items()
                                    if isinstance(vv, (int, float))}
                                for k, v in rows.items()}}

    b = out.get("baseline_all_background")
    if b:
        print(f"\nall-background baseline on test_factory: "
              f"pixel_accuracy={b['pixel_accuracy']:.4f} but clDice={b['cldice']:.4f}, "
              f"detect_rate={b['detect_rate']:.4f}")
        print("  -- a model predicting nothing scores that accuracy.")

    dst = BENCH / f"FULLEVAL_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {dst}")

if __name__ == "__main__":
    main()
