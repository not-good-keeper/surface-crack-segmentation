"""INT8 quantisation of an exported model, measured rather than assumed.

    python bench/quantize.py --model data/export/<name>.onnx

The deployment target is a CPU-only industrial PC already running the line software, so
size and single-core latency are product constraints, not vanity numbers. Static INT8
typically cuts a convolutional graph to about a quarter of its size and runs 2-3x
faster on CPU; what it costs in accuracy has to be measured on this model, on real
images, because a segmentation head is more sensitive to quantisation than a classifier
-- every pixel carries its own argmax, so a small logit shift moves a boundary.

Calibration uses images from the TRAIN split. Calibrating on the evaluation images would
tune the quantisation ranges to the data the result is then reported on, which is a
quiet way of leaking the test set into the deployed artefact.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_meta(path):
    import onnx
    g = onnx.load(str(path))
    return {e.key: e.value for e in g.metadata_props}


def tensor(name, size, prep, rgb):
    img = cv2.imread(str(ROOT / "data/clean/images" / f"{name}.png"), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if prep is not None:
        img = prep(img)
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
    a = arr.astype(np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None]


class Calib:
    """Feeds calibration batches; onnxruntime wants this exact interface."""

    def __init__(self, names, input_name, size, prep, rgb):
        self.names, self.input_name = names, input_name
        self.size, self.prep, self.rgb = size, prep, rgb
        self.i = 0

    def get_next(self):
        while self.i < len(self.names):
            x = tensor(self.names[self.i], self.size, self.prep, self.rgb)
            self.i += 1
            if x is not None:
                return {self.input_name: x}
        return None

    def rewind(self):
        self.i = 0


def bench(path, size, iters=40):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    n = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(5):
        sess.run(None, {n: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {n: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def agreement(fp32, int8, names, size, prep, rgb):
    """How often the two graphs assign a pixel the same class, and on defect pixels.

    Overall agreement is dominated by background, which both get right trivially. The
    number that matters is agreement where the fp32 model predicts a defect -- that is
    where a boundary shift becomes a changed finding.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    a = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
    na, nb = a.get_inputs()[0].name, b.get_inputs()[0].name
    same = tot = same_fg = tot_fg = 0
    for nm in names:
        x = tensor(nm, size, prep, rgb)
        if x is None:
            continue
        pa = a.run(None, {na: x})[0][0].argmax(0)
        pb = b.run(None, {nb: x})[0][0].argmax(0)
        same += int((pa == pb).sum()); tot += pa.size
        fg = pa > 0
        same_fg += int((pa[fg] == pb[fg]).sum()); tot_fg += int(fg.sum())
    return same / max(tot, 1), same_fg / max(tot_fg, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", type=int, default=200, help="train images for calibration")
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--per-channel", dest="per_channel", action="store_true", default=False)
    args = ap.parse_args()

    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType
    import onnxruntime as ort

    src = Path(args.model)
    meta = read_meta(src)
    size = int(meta.get("size", 256))
    rgb = str(meta.get("channel_order", "bgr")).lower() == "rgb"
    prep = None
    if meta.get("prep"):
        from preprocess import build
        prep = build(meta["prep"])
    print(f"{src.name}: size {size} | prep {meta.get('prep') or 'none'} | "
          f"{'RGB' if rgb else 'BGR'}")

    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    tr = df[df.split == "train"].sample(args.calib, random_state=0).name.tolist()
    ev = df[(df.split == "test_factory")].name.tolist()

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    # Quantisation rewrites the graph and drops metadata_props, so the contract the app
    # reads would be lost. Copied across explicitly below.
    prepped = src.with_name(src.stem + ".prep.onnx")
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)

    dst = src.with_name(src.stem + ".int8.onnx")
    quantize_static(str(prepped), str(dst),
                    Calib(tr, iname, size, prep, rgb),
                    quant_format=__import__("onnxruntime.quantization", fromlist=["QuantFormat"]).QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method={"minmax": CalibrationMethod.MinMax,
                                      "percentile": CalibrationMethod.Percentile,
                                      "entropy": CalibrationMethod.Entropy}[args.method],
                    per_channel=args.per_channel)
    prepped.unlink(missing_ok=True)

    import onnx
    g = onnx.load(str(dst))
    for k, v in meta.items():
        e = g.metadata_props.add(); e.key, e.value = k, str(v)
    onnx.save(g, str(dst))

    mb_a, mb_b = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    ma, pa_ = bench(src, size)
    mb, pb_ = bench(dst, size)
    ov, fg = agreement(src, dst, ev, size, prep, rgb)

    print(f"\n{'':<10}{'size MB':>9}{'median ms':>11}{'p90 ms':>9}")
    print(f"{'fp32':<10}{mb_a:>9.2f}{ma:>11.1f}{pa_:>9.1f}")
    print(f"{'int8':<10}{mb_b:>9.2f}{mb:>11.1f}{pb_:>9.1f}")
    print(f"{'ratio':<10}{mb_b/mb_a:>8.2f}x{mb/ma:>10.2f}x")
    print(f"\nagreement with fp32 on {len(ev)} test_factory images")
    print(f"  all pixels          {ov:.5f}")
    print(f"  where fp32 sees a defect  {fg:.5f}")
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
