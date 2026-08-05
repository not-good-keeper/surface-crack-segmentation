"""Per-class foreground thresholds, as ARCHITECTURE.md §7.1 specifies.

    python bench/class_thresh.py --tag V8A_11

`predict_split` uses a plain argmax, which has no background floor: a pixel scoring
0.34 crack / 0.33 scratch / 0.33 background comes out a confident crack. Detection is
0.96 while half of crack pixels are typed as scratch, so the boundary is the thing to
move, and this moves it without retraining.

Selection: among grid points holding any-defect clDice within `--cldice-tol` of the
argmax baseline, take the highest mean of crack and scratch recall. Geometry is what the
headline reports and must not be traded away; class recall is what is broken. Every grid
point is printed so a different rule can be applied by hand.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import runs                                 # noqa: E402
from data import CrackPatches               # noqa: E402
from metrics import evaluate_multiclass     # noqa: E402
from app.postprocess import Profile, class_map  # noqa: E402

BENCH = ROOT / "data/bench"
# Selection happens on val, full stop -- a threshold tuned on the split it is then
# reported on is a fit, not a measurement (ARCHITECTURE.md 7.1). Not a flag, so it
# cannot be pointed at a test split by accident.
SELECT_SPLIT = "val"
REPORT_SPLITS = ["test_factory", "test_factory_scratch", "test_negatives"]
GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
CLDICE_TOL = 0.01          # how far any-defect clDice may fall below argmax


def softmax_probs(model, ds, device, bs=32, max_batches=40):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    P, G = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(dl):
            if max_batches and bi >= max_batches:
                break
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x.to(device, non_blocking=True))
            P.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            G.append(y.numpy())
    return np.concatenate(P), np.concatenate(G)


def decide(P: np.ndarray, t_crack: float, t_scratch: float) -> np.ndarray:
    """§7.1 through the code the app ships -- a sweep run against a private copy of the
    decision logic would be tuning a model that never runs."""
    return class_map(P, Profile(crack_thresh=t_crack, scratch_thresh=t_scratch))


def score(P, G, tc, ts):
    """`tc is None` means the argmax baseline.

    Argmax is not reproducible by setting both thresholds very low: §7.1 only chooses
    between the foreground classes once one has passed, so thresholds at -1 make
    background unreachable. The baseline has to argmax over all three channels.
    """
    pred = P.argmax(axis=1) if tc is None else decide(P, tc, ts)
    r = evaluate_multiclass(pred, G)
    r.update(r["any_defect"])
    cr, sc = r.get("crack_class_recall"), r.get("scratch_class_recall")
    vals = [v for v in (cr, sc) if v is not None and not np.isnan(v)]
    return dict(cldice=r["cldice"], iou=r["iou"], det=r["detect_rate"],
                crack=cr, scratch=sc, balanced=float(np.mean(vals)) if vals else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-batches", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, ds_kw = runs.load(args.tag, device)

    P, G = softmax_probs(model,
                         CrackPatches(SELECT_SPLIT, train=False, **ds_kw),
                         device, max_batches=args.eval_batches)
    base = score(P, G, None, None)
    print(f"{args.tag}  select on '{SELECT_SPLIT}'  "
          f"(argmax baseline: clDice={base['cldice']:.4f} "
          f"crack={base['crack']:.3f} scratch={base['scratch']:.3f})\n")
    print(f"{'t_crack':>8}{'t_scratch':>10}{'clDice':>9}{'IoU':>8}{'det':>7}"
          f"{'crack':>8}{'scratch':>9}{'balanced':>10}")

    rows = []
    for tc in GRID:
        for ts in GRID:
            s = score(P, G, tc, ts)
            rows.append((tc, ts, s))
            print(f"{tc:>8.2f}{ts:>10.2f}{s['cldice']:>9.4f}{s['iou']:>8.4f}"
                  f"{s['det']:>7.3f}{s['crack']:>8.3f}{s['scratch']:>9.3f}"
                  f"{s['balanced']:>10.3f}")

    ok = [r for r in rows if r[2]["cldice"] >= base["cldice"] - CLDICE_TOL]
    if not ok:
        print("\nno grid point holds clDice within tolerance -- keeping argmax")
        return 0
    tc, ts, s = max(ok, key=lambda r: r[2]["balanced"])
    print(f"\nselected t_crack={tc:.2f} t_scratch={ts:.2f}: "
          f"balanced class recall {base['balanced']:.3f} -> {s['balanced']:.3f}, "
          f"clDice {base['cldice']:.4f} -> {s['cldice']:.4f}")

    print("\n--- applied to frozen test splits (thresholds NOT tuned on these) ---")
    for sp in REPORT_SPLITS:
        ds = CrackPatches(sp, train=False, **ds_kw)
        if not len(ds.pos) and not len(ds.neg):
            continue
        Pt, Gt = softmax_probs(model, ds, device, max_batches=args.eval_batches)
        b = score(Pt, Gt, None, None)
        n = score(Pt, Gt, tc, ts)
        print(f"{sp:<24} argmax: clD={b['cldice']:.4f} crack={b['crack']:.3f} "
              f"scratch={b['scratch']:.3f}")
        print(f"{'':<24} §7.1  : clD={n['cldice']:.4f} crack={n['crack']:.3f} "
              f"scratch={n['scratch']:.3f}")

if __name__ == "__main__":
    main()
