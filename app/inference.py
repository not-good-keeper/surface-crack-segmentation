"""ONNX inference: image bytes in, one inspection record out.

Kept separate from any view so the batch CLI, the tests and the UI run the identical
path. The UI is a prototype and may be replaced; this must not be.

Offline by construction (AC-7): nothing here opens a socket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import (Profile, build_record, class_map, extract_regions,  # noqa: E402
                         overlay, softmax)

ROOT = Path(__file__).resolve().parent.parent
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalise(img, prep=None):
    """One 256x256 BGR uint8 tile -> the tensor the graph expects.

    `prep` runs on the uint8 tile, before scaling, which is where bench/data.py puts it.
    """
    if prep is not None:
        img = prep(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def tile_origins(h, w, size=SIZE, stride=192):
    """Top-left corners covering (h,w), with the last row/column flush to the edge."""
    def axis(n):
        if n <= size:
            return [0]
        xs = list(range(0, n - size + 1, stride))
        if xs[-1] != n - size:
            xs.append(n - size)
        return xs
    return [(y, x) for y in axis(h) for x in axis(w)]


def to_source_scale(prob, shape):
    """(C,256,256) probabilities -> (C,h,w) at the resolution the caller submitted.

    A resized frame is inferred in 256-space, so every geometry the operator is shown --
    region length, maximum width, bounding box -- would otherwise be quoted in the units
    of an intermediate the operator never saw, and by a different factor on each axis for
    any non-square image. Interpolating the probabilities rather than the class map keeps
    the per-class thresholds of ARCHITECTURE 7.1 meaningful; upscaling an argmax would
    only round decisions already made at the wrong scale.

    This does not recover detail lost on the way down. It puts the answer in the right
    units, which is a separate thing from making it more accurate.
    """
    h, w = shape
    p = cv2.resize(prob.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    return p.transpose(2, 0, 1)


def preprocess(image_bgr, size=SIZE, prep=None):
    """Whole frame -> one tile, by resize. Retained for callers that want a single
    tensor; NOT the inspection path, because it rescales the defect.

    The training loader crops 256x256 at native scale and never resizes, so a frame
    downscaled to fit shows the model cracks several times thinner than any it was
    trained on. Measured on `test_factory` with V9_22: clDice 0.7196 -> 0.2556 and
    detection 0.949 -> 0.423 (T-02, bench/input_parity.py).
    """
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return normalise(img, prep)[None]


class Inspector:
    def __init__(self, model_path: str | Path, profile: Profile | None = None,
                 station_id: str = "unset"):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no model at {self.model_path}. Export one with "
                f"bench/export_onnx.py before running the app.")
        so = ort.SessionOptions()
        # Single thread: the deployment target is a CPU-only industrial PC that is also
        # running the line software, and a benchmark taken with every core free is not
        # a number anyone can plan against.
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.model_path), so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.classes = self.sess.get_outputs()[0].shape[1]
        self.profile = profile or Profile()
        self.station_id = station_id
        self.model_version = self.model_path.stem

        # The transform the weights were trained with, read from the graph rather than
        # passed in: a caller who swaps the .onnx should not have to know this changed.
        meta = self.sess.get_modelmeta().custom_metadata_map
        spec = meta.get("prep") or ""
        self.prep = None
        if spec:
            sys.path.insert(0, str(ROOT / "bench"))
            from preprocess import build
            self.prep = build(spec)
        self.prep_spec = spec

        # How the weights saw an image decides how we must feed them one. Trained on
        # whole frames scaled to 256 -> resize here; trained on native-scale crops ->
        # tile here. Getting this backwards costs about 0.45 clDice and shows up in no
        # metric, because both paths hand the graph the same tensor shape (T-02).
        self.resize_input = str(meta.get("resize", "1")).lower() in ("1", "true")

        # Input side, also read from the graph. A model trained at 512 fed 256-px tiles
        # sees every defect at half the width it learned, which is the same class of
        # silent failure as the crop/resize mismatch and just as invisible: both shapes
        # are valid input to a fully convolutional net. Older exports carry no `size`,
        # and 256 is what they were all trained at.
        try:
            self.size = int(meta.get("size", SIZE))
        except (TypeError, ValueError):
            self.size = SIZE

    def describe(self):
        """-> the contract this graph declares about how it must be fed.

        Read from the ONNX metadata rather than the filename. With more than one export
        in circulation the differences that matter -- 256 vs 512 input, which input
        transform, resize vs tile -- are invisible in a file listing, and picking the
        wrong one costs accuracy silently because every export accepts the same tensor
        shape. Surfaced through one method so the app, the batch CLI and the tests
        cannot describe the same file differently.
        """
        return {
            "file": self.model_path.name,
            "classes": self.classes,
            "input_size": self.size,
            "prep": self.prep_spec or "none",
            "input_mode": "resize whole frame" if self.resize_input
                          else f"tiled at native scale ({self.size} px tiles)",
        }

    def logits(self, image_bgr):
        """Single-tile logits by resize. Kept for the ONNX contract and for callers
        that genuinely want one 256x256 view; `probabilities` is the inspection path."""
        x = preprocess(image_bgr, size=self.size, prep=self.prep)
        return self.sess.run(None, {self.input_name: x})[0][0]     # (C,H,W)

    def probabilities(self, image_bgr, stride=192, batch=8):
        """Softmax probabilities at the source resolution, tiled at native scale.

        Defects here are 1-4 px wide, so scale is not a free parameter: resizing a
        1262 px frame to 256 thins a 4 px crack to under one pixel and the model --
        trained only on native-scale crops -- stops seeing it. Tiles overlap and
        overlapping predictions are averaged, which also removes the seam a defect
        crossing a tile boundary would otherwise get.
        """
        S = self.size
        stride = min(stride, S)
        h, w = image_bgr.shape[:2]
        pad_b, pad_r = max(S - h, 0), max(S - w, 0)
        if pad_b or pad_r:
            image_bgr = cv2.copyMakeBorder(image_bgr, 0, pad_b, 0, pad_r,
                                           cv2.BORDER_REFLECT)
        H, W = image_bgr.shape[:2]

        origins = tile_origins(H, W, S, stride)
        acc = np.zeros((3, H, W), np.float32)
        hits = np.zeros((H, W), np.float32)
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            x = np.stack([normalise(image_bgr[y:y + S, x0:x0 + S], self.prep)
                          for y, x0 in chunk])
            out = self.sess.run(None, {self.input_name: x})[0]
            for (y, x0), lg in zip(chunk, out):
                e = np.exp(lg - lg.max(axis=0, keepdims=True))
                acc[:, y:y + S, x0:x0 + S] += e / e.sum(axis=0, keepdims=True)
                hits[y:y + S, x0:x0 + S] += 1.0
        prob = acc / np.maximum(hits, 1.0)
        return prob[:, :h, :w]

    def inspect(self, image_bgr, image_bytes,
                product_id: str | None = None, material: str | None = None):
        """-> (record, overlay_bgr, class_map), all at the source resolution."""
        if self.classes == 1:
            # A binary checkpoint still runs, so the app is usable before the
            # three-class model finishes training -- but it can only ever report
            # cracks, and saying so here is better than silently labelling every
            # scratch a crack.
            lg = self.logits(image_bgr)
            p = 1.0 / (1.0 + np.exp(-lg))
            prob = np.concatenate([1.0 - p, p, np.zeros_like(p)], axis=0)
            prob = to_source_scale(prob, image_bgr.shape[:2])
        elif self.resize_input:
            prob = softmax(self.logits(image_bgr))
            prob = to_source_scale(prob, image_bgr.shape[:2])
        else:
            # Tiling already runs at native scale, so this branch is on source pixels.
            prob = self.probabilities(image_bgr)
        cmap = class_map(prob, self.profile)
        regions = extract_regions(cmap, self.profile)
        rec = build_record(image_bytes, regions, self.profile, self.model_version,
                           station_id=self.station_id, product_id=product_id,
                           material=material)
        return rec, overlay(image_bgr, cmap, regions), cmap
