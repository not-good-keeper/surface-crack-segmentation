"""Which input transform helps a trained checkpoint, and which pairs compose?

    python bench/prep_sweep.py --tag V9_22

Every op is scored against `none` on identical crops, selected on `val`, and only then
applied to the frozen test splits -- the same discipline bench/class_thresh.py follows,
for the same reason.

Read the result carefully. This measures test-time transforms against a model trained
without them, so a win means the op moves the input closer to what the model already
expects (usually by removing a nuisance factor). An op that loses here is not proven
useless: it may still pay off trained end to end, which this cannot tell you.
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
import preprocess as P                      # noqa: E402
from data import CrackPatches               # noqa: E402
from metrics import evaluate_multiclass     # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

# Selection is on val and nowhere else. Not a flag, so it cannot be aimed at a test
# split by accident -- a transform tuned on the split it is reported on is a fit.
SELECT_SPLIT = "val"
REPORT_SPLITS = ["test_factory", "test_unseen_material", "test_negatives"]
BATCHES = 25               # 800 images; enough to rank, cheap enough to sweep 22 configs
TOP = 4                    # how many val winners are carried to the frozen splits


def cache(split, ds_kw, batches, bs=32):
    """-> (N,H,W,3) uint8 BGR patches and their targets.

    Pulled back through the dataset's own normalisation rather than re-reading the
    manifest: evaluation crops are seeded off the sample index, so this is the only way
    to be sure every op sees pixel-identical inputs.
    """
    dl = DataLoader(CrackPatches(split, train=False, **ds_kw),
                    batch_size=bs, shuffle=False, num_workers=2)
    imgs, gts = [], []
    for bi, (x, y) in enumerate(dl):
        if batches and bi >= batches:
            break
        a = x.numpy().transpose(0, 2, 3, 1) * STD + MEAN
        imgs.append(np.clip(a, 0, 255).astype(np.uint8))
        gts.append(y.numpy().astype(np.uint8))
    return np.concatenate(imgs), np.concatenate(gts)


def infer(model, imgs, fn, device, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            b = np.stack([fn(im) for im in imgs[i:i + bs]]).astype(np.float32)
            b = ((b - MEAN) / STD).transpose(0, 3, 1, 2)
            x = torch.from_numpy(np.ascontiguousarray(b)).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out.append(model(x).float().argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def score(pred, gt):
    r = evaluate_multiclass(pred, gt.astype(np.int64))
    a = r["any_defect"]
    return dict(cldice=a["cldice"], iou=a["iou"], det=a["detect_rate"],
                fp=a["fp_area"], crack=r["crack_class_recall"],
                scratch=r["scratch_class_recall"])


def row(name, s, base=None):
    d = "" if base is None else f"{s['cldice'] - base['cldice']:+7.4f}"
    return (f"{name:<26}{s['cldice']:>8.4f}{d:>9}{s['iou']:>8.4f}{s['det']:>7.3f}"
            f"{s['crack']:>8.3f}{s['scratch']:>9.3f}{s['fp']:>9.5f}")


HEAD = (f"{'op':<26}{'clDice':>8}{'delta':>9}{'IoU':>8}{'det':>7}"
        f"{'crack':>8}{'scratch':>9}{'fp_area':>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="stored run; omit when using --weights")
    ap.add_argument("--weights", help="a .pt / .best.pt from a run still in flight")
    ap.add_argument("--model", default="smpslim_timm-mobilenetv3_small_100")
    ap.add_argument("--resize", action="store_true",
                    help="score under the resize regime (must match how it trained)")
    ap.add_argument("--ops", nargs="*", help="restrict the grid; 'none' is always added")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.weights:
        model, _, ds_kw = runs.load_weights(args.weights, args.model, 3, device,
                                            camera_profile="conveyor",
                                            resize=args.resize)
    else:
        model, _, ds_kw = runs.load(args.tag, device)

    specs = (["none"] + [o for o in args.ops if o != "none"] if args.ops
             else list(P.OPS) + ["+".join(p) for p in P.PAIRS])
    imgs, gts = cache(SELECT_SPLIT, ds_kw, BATCHES)
    print(f"{args.tag or Path(args.weights).stem}  select on '{SELECT_SPLIT}'  n={len(imgs)}\n")
    print(HEAD)

    base, results = None, []
    for spec in specs:
        s = score(infer(model, imgs, P.build(spec), device), gts)
        if spec == "none":
            base = s
        results.append((spec, s))
        print(row(spec, s, None if spec == "none" else base), flush=True)

    del imgs, gts
    winners = [r for r in results if r[0] != "none"]
    winners.sort(key=lambda r: -r[1]["cldice"])
    winners = [w for w in winners[:TOP] if w[1]["cldice"] > base["cldice"]]
    if not winners:
        print("\nno transform beats the raw input on val -- nothing to carry forward")
        return 0
    print("\ncarrying forward: " + ", ".join(w[0] for w in winners))

    for sp in REPORT_SPLITS:
        imgs, gts = cache(sp, ds_kw, BATCHES)
        if not len(imgs):
            continue
        print(f"\n--- {sp} (n={len(imgs)}, transforms NOT tuned here) ---")
        print(HEAD)
        b = score(infer(model, imgs, P.build("none"), device), gts)
        print(row("none", b))
        for spec, _ in winners:
            print(row(spec, score(infer(model, imgs, P.build(spec), device), gts), b),
                  flush=True)
        del imgs, gts
    return 0


if __name__ == "__main__":
    sys.exit(main())
