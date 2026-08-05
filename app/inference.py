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


def preprocess(image_bgr, size=SIZE, prep=None):
    """Deterministic resize + optional input transform + ImageNet normalisation,
    matching bench/data.py exactly.

    Plain resize rather than letterbox: the training loader resizes, and preprocessing
    that differs between training and deployment is a silent accuracy loss that no
    metric in the benchmark would reveal. `prep` is the same clause -- it runs on the
    resized BGR uint8 frame, which is where bench/data.py applies it.
    """
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if prep is not None:
        img = prep(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


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
        self.profile = profile or Profile()
        self.station_id = station_id
        self.model_version = self.model_path.stem

        # The transform the weights were trained with, read from the graph rather than
        # passed in: a caller who swaps the .onnx should not have to know this changed.
        spec = self.sess.get_modelmeta().custom_metadata_map.get("prep") or ""
        self.prep = None
        if spec:
            sys.path.insert(0, str(ROOT / "bench"))
            from preprocess import build
            self.prep = build(spec)
        self.prep_spec = spec

    def logits(self, image_bgr):
        x = preprocess(image_bgr, prep=self.prep)
        return self.sess.run(None, {self.input_name: x})[0][0]     # (C,H,W)

    def inspect(self, image_bgr, image_bytes,
                product_id: str | None = None, material: str | None = None):
        """-> (record, overlay_bgr, class_map)."""
        lg = self.logits(image_bgr)
        if lg.shape[0] == 1:
            # A binary checkpoint still runs, so the app is usable before the
            # three-class model finishes training -- but it can only ever report
            # cracks, and saying so here is better than silently labelling every
            # scratch a crack.
            p = 1.0 / (1.0 + np.exp(-lg))
            prob = np.concatenate([1.0 - p, p, np.zeros_like(p)], axis=0)
        else:
            prob = softmax(lg)
        cmap = class_map(prob, self.profile)
        regions = extract_regions(cmap, self.profile)
        rec = build_record(image_bytes, regions, self.profile, self.model_version,
                           station_id=self.station_id, product_id=product_id,
                           material=material)
        return rec, overlay(image_bgr, cmap, regions), cmap
