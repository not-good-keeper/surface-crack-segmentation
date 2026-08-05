"""Input transforms, swept by bench/prep_sweep.py.

Everything here takes and returns the BGR uint8 patch, before ImageNet normalisation,
so whatever wins the sweep drops into bench/data.py unchanged.

Ops are registered in OPS and composed by name: `chain("flatten", "clahe2")`.
"""
from __future__ import annotations

from functools import partial

import cv2
import numpy as np


def clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def flatten(img, sigma=31):
    """Divide out the illumination field.

    A ring light over a belt leaves a fixed bright centre that carries no information
    about the product. The defect is the residual, not the gradient.
    """
    bg = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    return np.clip(img / np.maximum(bg, 1.0) * 128.0, 0, 255).astype(np.uint8)


def blackhat(img, k=9, gain=1.0):
    """Deepen dark thin structures -- what a crack is on almost every surface here."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se).astype(np.int16)
    return np.clip(img - gain * bh, 0, 255).astype(np.uint8)


def tophat(img, k=9, gain=1.0):
    """Brighten bright thin structures -- a polished-metal scratch catches the light."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se).astype(np.int16)
    return np.clip(img + gain * th, 0, 255).astype(np.uint8)


def bothat(img, k=9, gain=1.0):
    """Both at once: cracks read dark, scratches read bright, and the model has to
    separate them. Pushing each away from the surface exaggerates that difference."""
    return tophat(blackhat(img, k, gain), k, gain)


def unsharp(img, sigma=2.0, amount=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def gamma(img, g=0.7):
    lut = np.clip((np.arange(256) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def median(img, k=3):
    return cv2.medianBlur(img, k)


def bilateral(img, d=5, sc=50, ss=50):
    return cv2.bilateralFilter(img, d, sc, ss)


def gauss(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def nlmeans(img, h=3):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


OPS = {
    "none": lambda img: img,
    "clahe2": partial(clahe, clip=2.0),
    "clahe4": partial(clahe, clip=4.0),
    "flatten": flatten,
    "blackhat": blackhat,
    "tophat": tophat,
    "bothat": bothat,
    "unsharp": unsharp,
    "unsharp_soft": partial(unsharp, sigma=1.5, amount=0.5),
    "gamma07": partial(gamma, g=0.7),
    "gamma13": partial(gamma, g=1.3),
    "median3": median,
    "median5": partial(median, k=5),
    "bilateral": bilateral,
    "bilateral_hard": partial(bilateral, d=7, sc=75, ss=75),
    "gauss1": gauss,
    "nlmeans": nlmeans,
}

# Ordered deliberately: denoise before sharpening, flatten before contrast. Reversing
# either amplifies what the first step was there to remove.
PAIRS = [
    ("flatten", "clahe2"),
    ("flatten", "blackhat"),
    ("flatten", "bothat"),
    ("median3", "unsharp_soft"),
    ("bilateral", "clahe2"),
    ("clahe2", "unsharp_soft"),
    ("median3", "clahe2"),
    ("clahe2", "blackhat"),
    ("flatten", "clahe2", "unsharp_soft"),
]


def chain(*names):
    fns = [OPS[n] for n in names]

    def run(img):
        for f in fns:
            img = f(img)
        return img
    return run


def build(spec):
    """`spec` is a '+'-joined op name, e.g. 'flatten+clahe2'."""
    return chain(*spec.split("+"))
