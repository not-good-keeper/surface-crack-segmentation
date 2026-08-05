"""Segmentation metrics appropriate for thin curvilinear structures.

Plain IoU is the standard reported number but it is badly misleading here. On a 2 px
wide crack a one-pixel localisation shift removes most of the intersection and can
halve IoU without any meaningful error. Two extra metrics fix that:

  clDice        centreline Dice. Measures whether the predicted structure follows the
                same path and stays connected, which is what actually matters for a
                crack -- a prediction broken into dashes is far worse than one that is
                slightly too thick, and IoU cannot tell those apart.
  tolerant F1   allows `tol` px of localisation slack, so we measure "did you find the
                crack" separately from "did you place it to the exact pixel".

And a false-positive metric, because the deployment failure is a phone app flagging
every joint and grain line:

  fp_area       predicted foreground fraction on images with NO crack. At ~2-4 %
                foreground, an all-background predictor gets 96-98 % pixel accuracy,
                so accuracy-like metrics must never be reported alone.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


# ------------------------------------------------------------------ torch, batched
@torch.no_grad()
def confusion(pred: torch.Tensor, gt: torch.Tensor):
    p = pred.bool()
    g = gt.bool()
    tp = (p & g).sum(dim=(1, 2, 3)).float()
    fp = (p & ~g).sum(dim=(1, 2, 3)).float()
    fn = (~p & g).sum(dim=(1, 2, 3)).float()
    return tp, fp, fn


@torch.no_grad()
def iou_dice(pred, gt, eps=1e-7):
    tp, fp, fn = confusion(pred, gt)
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return iou, dice


# ------------------------------------------------------------------ numpy, per-image
def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    if not mask.any():
        return np.zeros_like(mask, bool)
    return skeletonize(mask.astype(bool))


def cl_dice(pred: np.ndarray, gt: np.ndarray, eps=1e-7) -> float:
    """Topology-aware Dice: how well each skeleton lies inside the other mask."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = _skeleton(p), _skeleton(g)
    t_prec = (sp & g).sum() / (sp.sum() + eps)   # is the prediction on the true path
    t_sens = (sg & p).sum() / (sg.sum() + eps)   # did we cover the true centreline
    return float(2 * t_prec * t_sens / (t_prec + t_sens + eps))


def tolerant_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2, eps=1e-7) -> float:
    """F1 allowing `tol` px of localisation error on both sides."""
    p, g = pred.astype(np.uint8), gt.astype(np.uint8)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(g, k)
    pd_ = cv2.dilate(p, k)
    prec = (p.astype(bool) & gd.astype(bool)).sum() / (p.sum() + eps)
    rec = (g.astype(bool) & pd_.astype(bool)).sum() / (g.sum() + eps)
    return float(2 * prec * rec / (prec + rec + eps))


def evaluate(preds: np.ndarray, gts: np.ndarray, tol: int = 2) -> dict:
    """preds/gts: (N,H,W) uint8/bool. Splits positives and negatives correctly."""
    ious, dices, clds, tf1s = [], [], [], []
    fp_area, fp_hit, n_neg = [], 0, 0
    det_hit, n_pos = 0, 0

    for p, g in zip(preds, gts):
        p = p.astype(bool)
        g = g.astype(bool)
        if not g.any():
            # negative image: the only meaningful numbers are false-positive ones
            n_neg += 1
            fp_area.append(p.mean())
            fp_hit += int(p.sum() > 0.0005 * p.size)
            continue
        n_pos += 1
        inter = (p & g).sum()
        union = (p | g).sum()
        ious.append(inter / max(union, 1))
        dices.append(2 * inter / max(p.sum() + g.sum(), 1))
        clds.append(cl_dice(p, g))
        tf1s.append(tolerant_f1(p, g, tol))
        det_hit += int(inter > 0)

    m = lambda a: float(np.mean(a)) if len(a) else float("nan")  # noqa: E731
    return {
        "iou": m(ious), "dice": m(dices), "cldice": m(clds),
        f"tf1@{tol}": m(tf1s),
        "detect_rate": det_hit / max(n_pos, 1),
        "fp_area": m(fp_area),                 # mean predicted fg on clean images
        "fp_image_rate": fp_hit / max(n_neg, 1),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def evaluate_multiclass(preds: np.ndarray, gts: np.ndarray, tol: int = 2,
                        names=("background", "crack", "scratch")) -> dict:
    """preds/gts: (N,H,W) int index maps. Per-class metrics plus class confusion.

    Each class is reduced to its own binary problem so the numbers stay comparable
    with the binary results. Confusion is over defect pixels only -- including
    background would swamp the class question with the easy majority.
    """
    out = {}
    for ci, cname in enumerate(names):
        if ci == 0:
            continue
        out[cname] = evaluate(preds == ci, gts == ci, tol)

    n_cls = len(names)
    cm = np.zeros((n_cls, n_cls), np.int64)
    for p, g in zip(preds, gts):
        m = g > 0                                   # true defect pixels only
        if not m.any():
            continue
        np.add.at(cm, (g[m].ravel(), p[m].ravel()), 1)
    out["confusion_defect_px"] = cm.tolist()
    out["class_names"] = list(names)

    # Per-class recall over defect pixels: of the pixels truly of class c, what
    # fraction did the model call c? This is the TC-03 number.
    for ci, cname in enumerate(names):
        if ci == 0:
            continue
        tot = int(cm[ci].sum())
        out[f"{cname}_class_recall"] = float(cm[ci, ci] / tot) if tot else float("nan")

    # Foreground-vs-background agreement, independent of which defect type was chosen.
    # A model can be excellent at finding defects and bad at naming them; separating
    # these two says which problem to work on.
    out["any_defect"] = evaluate(preds > 0, gts > 0, tol)
    return out


def all_background_baseline(gts: np.ndarray) -> dict:
    """Sanity floor. Proves the metric set cannot be gamed by predicting nothing:
    pixel accuracy will look excellent while clDice and detect_rate are 0."""
    zeros = np.zeros_like(gts)
    r = evaluate(zeros, gts)
    acc = float(np.mean([(g == 0).mean() for g in gts]))
    r["pixel_accuracy"] = acc
    return r
