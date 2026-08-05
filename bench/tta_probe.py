"""Does test-time augmentation buy anything, and is it worth the latency?

    python bench/tta_probe.py --weights data/bench/<ckpt>.best.pt

Four inference rules on one checkpoint, scored on the deployed path (resize, then
upsample to source): plain, flip-averaged, multi-scale, and both.

Multi-scale is the interesting one. The model is fully convolutional, so it accepts 384
without retraining, and the whole reason the resize path scores low is that a 4 px crack
at 793 px becomes ~1.3 px at 256. If the network generalises to a larger input, some of
that is recoverable tonight rather than in a retrain. It may equally degrade -- batch-norm
statistics and the effective receptive field were both fixed at 256 -- which is why this
measures instead of assuming.

Latency is reported beside accuracy because every rule here multiplies inference cost,
and the deployment target is a CPU box already running the line software.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import runs                                       # noqa: E402
from data import ROLE_CLASS, BACKGROUND           # noqa: E402
from metrics import evaluate_multiclass           # noqa: E402
from preprocess import build                      # noqa: E402

CLEAN = ROOT / "data/clean"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_pair(name, role):
    img = cv2.imread(str(CLEAN / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(CLEAN / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    if msk is None:
        msk = np.zeros(img.shape[:2], np.uint8)
    y = np.zeros(msk.shape, np.int64)
    y[msk > 127] = ROLE_CLASS.get(str(role), BACKGROUND)
    return img, y


def tensor(img, size, prep):
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if prep is not None:
        x = prep(x)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def probs_at(model, img, size, prep, device, flips):
    """Softmax at `size`, optionally averaged over flips, returned at `size`."""
    x = torch.from_numpy(tensor(img, size, prep)).to(device)
    views = [(x, None)]
    if flips:
        views += [(torch.flip(x, [3]), [3]), (torch.flip(x, [2]), [2])]
    acc = None
    with torch.no_grad():
        for v, undo in views:
            p = torch.softmax(model(v).float(), dim=1)
            if undo:
                p = torch.flip(p, undo)
            acc = p if acc is None else acc + p
    return (acc / len(views))[0].cpu().numpy()


def score(model, names, roles, prep, device, sizes, flips):
    preds, gts = [], []
    for name, role in zip(names, roles):
        img, y = load_pair(name, role)
        if img is None:
            continue
        h, w = img.shape[:2]
        acc = None
        for s in sizes:
            p = probs_at(model, img, s, prep, device, flips)
            # Every scale is brought to the source resolution before averaging, since
            # that is where the answer is finally read and the scales differ in size.
            up = cv2.resize(p.transpose(1, 2, 0), (w, h),
                            interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
            acc = up if acc is None else acc + up
        preds.append((acc / len(sizes)).argmax(0).astype(np.int64))
        gts.append(y)
    keys = ("cldice", "iou", "detect_rate", "tf1@2")
    rows = []
    for p, g in zip(preds, gts):
        r = evaluate_multiclass(p[None], g[None])
        r.update(r["any_defect"])
        rows.append([r.get(k, np.nan) for k in keys])
    at_source = dict(zip(keys, np.nanmean(np.array(rows, float), axis=0)))

    # The same predictions scored the way bench/train.py scores them: both sides shrunk
    # to 256. Nearest-neighbour downscaling of a 1-4 px crack by 3-5x leaves a short,
    # broken GT skeleton, and clDice has far less to disagree with -- so this reads
    # higher without the model being any better. The deployed system emits a mask at
    # source resolution and is judged against the real label, which is the other row.
    rows = []
    for p, g in zip(preds, gts):
        ps = cv2.resize(p.astype(np.uint8), (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(np.int64)
        gs = cv2.resize(g.astype(np.uint8), (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(np.int64)
        r = evaluate_multiclass(ps[None], gs[None])
        r.update(r["any_defect"])
        rows.append([r.get(k, np.nan) for k in keys])
    at_256 = dict(zip(keys, np.nanmean(np.array(rows, float), axis=0)))
    return at_source, at_256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="smpslim_timm-mobilenetv3_small_100")
    ap.add_argument("--prep", default="bilateral")
    ap.add_argument("--split", default="test_factory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = runs.load_weights(args.weights, args.model, 3, device)
    prep = build(args.prep) if args.prep else None

    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    df = df[(df.split == args.split) & df.role.isin(["positive", "scratch"])
            & (df.crack_px > 0)]
    names, roles = df.name.tolist(), df.role.tolist()
    print(f"{Path(args.weights).name}  {args.split}  n={len(names)}  prep={args.prep}\n")
    print(f"{'rule':<22}{'scoring':>11}{'clDice':>9}{'IoU':>9}{'tf1@2':>9}"
          f"{'det':>8}{'s/img':>8}")

    base = None
    for label, sizes, flips in (("plain 256", (256,), False),
                                ("flips 256", (256,), True),
                                ("multi-scale 256+384", (256, 384), False),
                                ("multi-scale + flips", (256, 384), True)):
        t0 = time.time()
        per_image, pooled = score(model, names, roles, prep, device, sizes, flips)
        per = (time.time() - t0) / max(len(names), 1)
        d = "" if base is None else f"  ({per_image['cldice'] - base:+.4f})"
        base = per_image["cldice"] if base is None else base
        for how, r in (("@source", per_image), ("@256 GT", pooled)):
            tail = f"{per:>8.3f}{d}" if how == "@source" else ""
            print(f"{label if how == '@source' else '':<22}{how:>11}"
                  f"{r['cldice']:>9.4f}{r['iou']:>9.4f}{r['tf1@2']:>9.4f}"
                  f"{r['detect_rate']:>8.3f}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
