"""Losses for thin-structure segmentation.

The headline metric is clDice -- agreement of centrelines, not of areas -- but training
optimises cross-entropy plus area Dice, and neither is told that a crack is a connected
curve. A prediction broken into three fragments scores almost the same area Dice as an
unbroken one and far worse clDice, so the loss is indifferent to exactly the property
being reported.

`soft_cldice` closes that gap. It is the differentiable relaxation from Shit et al.,
"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
(CVPR 2021): morphological skeletonisation rewritten with min/max pooling so it carries
a gradient, then the same harmonic mean of centreline precision and sensitivity that the
metric uses.

Used as a term added to the existing loss rather than a replacement. On its own it is
happy with a one-pixel-wide thread through the middle of a wide defect, which would
destroy the width measurements the product reports.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_erode(x):
    """Greyscale erosion by a 3x3 cross, as a differentiable min-pool."""
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iters: int = 3):
    """Differentiable medial axis by iterative thinning.

    Each pass peels one layer and keeps whatever the opening cannot restore -- the
    ridge. `iters` bounds the half-width of a structure that can be fully thinned, so
    3 covers the 1-7 px defects in this corpus; larger values cost time and gradient
    quality without reaching anything wider that we care about.
    """
    x1 = soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x)
        x1 = soft_open(x)
        delta = F.relu(x - x1)
        # `skel * delta` subtracts what is already counted, so overlapping passes do
        # not accumulate mass at the centre.
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(prob_fg, target_fg, iters: int = 3, eps: float = 1.0):
    """1 - clDice between a foreground probability map and its binary target.

    Both (N,H,W), values in [0,1]. Mirrors the reported metric: `tprec` is how much of
    the predicted centreline lies inside the true mask, `tsens` how much of the true
    centreline lies inside the prediction, combined harmonically so neither can be
    bought by sacrificing the other.
    """
    sk_p = soft_skeleton(prob_fg.unsqueeze(1), iters).squeeze(1)
    sk_t = soft_skeleton(target_fg.unsqueeze(1), iters).squeeze(1)
    tprec = (torch.sum(sk_p * target_fg) + eps) / (torch.sum(sk_p) + eps)
    tsens = (torch.sum(sk_t * prob_fg) + eps) / (torch.sum(sk_t) + eps)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def cldice_term(logits, y, classes_fg=(1, 2), iters: int = 3):
    """Mean soft-clDice over the foreground classes of a multi-class head.

    Background is excluded for the same reason it is excluded from the area Dice term:
    its centreline is meaningless and its mass would drown the classes that matter.
    Classes absent from a batch contribute nothing rather than a spurious perfect score.
    """
    p = torch.softmax(logits, dim=1)
    total, n = logits.new_zeros(()), 0
    for c in classes_fg:
        t = (y == c).float()
        if t.sum() == 0:
            continue
        total = total + soft_cldice(p[:, c], t, iters=iters)
        n += 1
    return total / max(n, 1)
