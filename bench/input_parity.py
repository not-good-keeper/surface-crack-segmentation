"""T-02: does the deployed input path match the one the metrics were measured on?

    python bench/input_parity.py --tag V9_22

`bench/data.py` takes a 256x256 crop at native scale. `app/inference.py` resizes the
whole frame to 256x256. Both produce a (3,256,256) tensor, so nothing downstream can
tell them apart -- and every reported number comes from the crop path while every
deployed inference comes from the resize path.

This compares them on the same images and the same weights. A gap here is not a model
result; it is the benchmark measuring something the product does not do.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import runs                                  # noqa: E402
import pandas as pd                          # noqa: E402
from data import CrackPatches, ROLE_CLASS, BACKGROUND   # noqa: E402
from metrics import evaluate_multiclass      # noqa: E402
from preprocess import build                 # noqa: E402
sys.path.insert(0, str(ROOT / 'app'))
from inference import tile_origins            # noqa: E402

CLEAN = ROOT / "data/clean"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 256


def load_pair(name, role, classes):
    img = cv2.imread(str(CLEAN / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(CLEAN / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    if msk is None:
        msk = np.zeros(img.shape[:2], np.uint8)
    y = np.zeros(msk.shape, np.int64)
    y[msk > 127] = ROLE_CLASS.get(str(role), BACKGROUND) if classes > 1 else 1
    return img, y


def resize_batch(names, roles, prep, classes):
    """The old app path: whole frame -> 256, mask resized with it."""
    X, Y = [], []
    for name, role in zip(names, roles):
        img, y = load_pair(name, role, classes)
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(y.astype(np.uint8), (SIZE, SIZE),
                       interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if prep is not None:
            img = prep(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        X.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        Y.append(y)
    return np.stack(X), np.stack(Y)


def deployed_score(model, names, roles, prep, classes, device):
    """The app path for a resize-trained model: infer at 256, report at source scale.

    `resize_batch` shrinks the mask to meet the prediction, which is how training scores
    itself. The operator is not shown a 256x256 answer -- app/inference.py interpolates
    the probabilities back up, and every region length and width is measured there. So
    this scores the upsampled prediction against the mask at its own resolution, which is
    the only variant of the resize path the product actually produces.

    It should read lower than the 256-space number: detail lost on the way down does not
    come back, and clDice is sensitive to exactly that on 1-4 px structures. The size of
    that drop is the cost of the resize design, and it belongs in the open.
    """
    preds, gts = [], []
    for name, role in zip(names, roles):
        img, y = load_pair(name, role, classes)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        if prep is not None:
            small = prep(small)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        with torch.no_grad():
            lg = model(torch.from_numpy(x).to(device)).float().cpu().numpy()[0]
        e = np.exp(lg - lg.max(axis=0, keepdims=True))
        prob = e / e.sum(axis=0, keepdims=True)
        up = cv2.resize(prob.transpose(1, 2, 0), (w, h),
                        interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        preds.append(up.argmax(0).astype(np.int64))
        gts.append(y)
    return preds, gts


def tiled_score(model, names, roles, prep, classes, device, stride=192):
    """The current app path: tiles at native scale, averaged over overlaps.

    Scored at full resolution against the unresized mask, which is the geometry the
    operator is shown -- not a 256x256 proxy for it.
    """
    preds, gts = [], []
    for name, role in zip(names, roles):
        img, y = load_pair(name, role, classes)
        if img is None:
            continue
        h, w = img.shape[:2]
        pb, pr = max(SIZE - h, 0), max(SIZE - w, 0)
        if pb or pr:
            img = cv2.copyMakeBorder(img, 0, pb, 0, pr, cv2.BORDER_REFLECT)
        H, W = img.shape[:2]
        acc = np.zeros((classes, H, W), np.float32)
        hits = np.zeros((H, W), np.float32)
        for y0, x0 in tile_origins(H, W, SIZE, stride):
            t = img[y0:y0 + SIZE, x0:x0 + SIZE]
            if prep is not None:
                t = prep(t)
            rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
            with torch.no_grad():
                lg = model(torch.from_numpy(x).to(device)).float().cpu().numpy()[0]
            e = np.exp(lg - lg.max(axis=0, keepdims=True))
            acc[:, y0:y0 + SIZE, x0:x0 + SIZE] += e / e.sum(axis=0, keepdims=True)
            hits[y0:y0 + SIZE, x0:x0 + SIZE] += 1.0
        prob = (acc / np.maximum(hits, 1.0))[:, :h, :w]
        preds.append(prob.argmax(0).astype(np.int64))
        gts.append(y)
    return preds, gts


def mean_metrics(preds, gts):
    """Per-image metrics averaged over images.

    The three paths produce different resolutions, so a pooled score would weight a
    1262 px frame far above a 256 px one purely because it has more pixels. Averaging
    per image makes them comparable.
    """
    keys = ("cldice", "iou", "detect_rate", "crack_class_recall")
    rows = []
    for p, g in zip(preds, gts):
        r = evaluate_multiclass(p[None], g[None])
        r.update(r["any_defect"])
        rows.append([r.get(k, np.nan) for k in keys])
    a = np.array(rows, float)
    return dict(zip(keys, np.nanmean(a, axis=0)))


def infer_batch(model, X, device, bs=32):
    P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                P.append(model(x).float().argmax(1).cpu().numpy())
    return list(np.concatenate(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test_factory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run, ds_kw = runs.load(args.tag, device)
    prep = build(ds_kw["prep"]) if ds_kw.get("prep") else None

    # crop -- exactly what every stored metric was measured on
    ds = CrackPatches(args.split, train=False, **ds_kw)
    Xc, Yc = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        Xc.append(x.numpy())
        Yc.append(y.numpy())
    crop = mean_metrics(infer_batch(model, np.stack(Xc), device), Yc)

    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    df = df[df.split == args.split]
    df = df[df.role.isin(["positive", "scratch"]) & (df.crack_px > 0)]
    names, roles = df.name.tolist(), df.role.tolist()

    # resize -- the app path before tiling
    Xr, Yr = resize_batch(names, roles, prep, ds_kw["classes"])
    rez = mean_metrics(infer_batch(model, Xr, device), list(Yr))

    # tiled at native scale, and resize-then-upsample: the two candidate app paths
    tp, tg = tiled_score(model, names, roles, prep, ds_kw["classes"], device)
    tiled = mean_metrics(tp, tg)
    dp, dg = deployed_score(model, names, roles, prep, ds_kw["classes"], device)
    upsampled = mean_metrics(dp, dg)

    # Which row is the claim and which is the product depends on how the run was
    # trained, so neither is hard-coded. A resize-trained run reports the 256-space
    # resize number and deploys the upsampled one; a crop-trained run reports the crop
    # number and deploys tiles. Comparing the wrong pair is how the original mismatch
    # went unseen for nine runs.
    resized_run = bool(ds_kw.get("resize"))
    ref_name, ref = (("resize @256 (reported)", rez) if resized_run
                     else ("crop @native (reported)", crop))
    dep_name, dep = (("resize -> upsample (app)", upsampled) if resized_run
                     else ("tiled @native (app)", tiled))

    print(f"{args.tag}  {args.split}  prep={ds_kw.get('prep') or 'none'}  "
          f"resize_run={resized_run}  n={len(names)}\n")
    print(f"{'path':<28}{'clDice':>9}{'IoU':>9}{'det':>8}{'crackR':>9}")
    for name, r in (("crop @native", crop),
                    ("resize @256", rez),
                    ("resize -> upsample", upsampled),
                    ("tiled @native", tiled)):
        mark = "  <- reported" if r is ref else ("  <- deployed" if r is dep else "")
        print(f"{name:<28}{r['cldice']:>9.4f}{r['iou']:>9.4f}"
              f"{r['detect_rate']:>8.3f}{r['crack_class_recall']:>9.3f}{mark}")

    gap = dep["cldice"] - ref["cldice"]
    print(f"\nreported = {ref_name}\ndeployed = {dep_name}")
    print(f"deployed vs reported: clDice {gap:+.4f}   "
          f"detection {dep['detect_rate'] - ref['detect_rate']:+.3f}")
    if abs(gap) > 0.05:
        print("\nFAIL (T-02): the deployed path does not reproduce the measured path. "
              "Every number quoted for this model describes an input the product does "
              "not produce.")
        return 1
    print("\nPASS (T-02)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
