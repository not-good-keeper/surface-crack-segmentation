"""ONNX export + latency measurement on a quiet machine.

Latency is half the deployment decision and it is the number most easily corrupted:
during this project the same `tiny_unet` measured 96 ms, 103 ms and 412 ms depending on
what else was running. This script therefore refuses to report a number while other
heavy Python processes are alive, rather than silently producing a meaningless one.

ONNX Runtime on desktop CPU is the honest proxy available here. A mid-range phone is
typically 3-6x slower than a desktop core, so treat these figures as an optimistic
lower bound, not a phone measurement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/export"
SIZE = 256          # fixed by the tensor contract in docs/INTEGRATION.md


def machine_busy(threshold: int = 3) -> int:
    """Count heavy python processes other than this one."""
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                           capture_output=True, text=True, timeout=30)
        n = max(0, len(r.stdout.strip().splitlines()) - 1)
        return n
    except Exception:  # noqa: BLE001
        return 0


def bench_torch_cpu(model, size=256, iters=20):
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def bench_onnx(path, size=256, iters=30):
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # single core = the phone-like case
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def check_parity(model, path, size=256, classes=1, n=4):
    """Confirm the exported graph reproduces the torch model.

    Opset lowering and graph optimisation can shift numerics, and a segmentation head
    is more sensitive than a classifier -- every pixel is its own argmax.
    `max_abs_logit_diff` is drift; `argmax_agreement` is the fraction of pixels that
    still get the same class, which is the one that matters operationally.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    model = model.to("cpu").eval()
    diffs, agree = [], []
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.standard_normal((1, 3, size, size)).astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).numpy()
        o = sess.run(None, {iname: x})[0]
        diffs.append(float(np.abs(t - o).max()))
        if classes > 1:
            agree.append(float((t.argmax(1) == o.argmax(1)).mean()))
        else:
            agree.append(float(((t > 0) == (o > 0)).mean()))
    return dict(max_abs_logit_diff=max(diffs), argmax_agreement=min(agree))


def stamp(path, **meta):
    """Record how the weights were trained inside the graph itself.

    A sidecar file gets separated from the model it describes; metadata does not. The
    app reads `prep` back out, so swapping in a checkpoint trained with a different
    input transform cannot silently mismatch the deployment path.
    """
    import onnx
    g = onnx.load(str(path))
    for k, v in meta.items():
        e = g.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(g, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--weights", default=None, help="optional .pt state_dict for one model")
    ap.add_argument("--classes", type=int, default=1,
                    help="1 = binary crack head, 3 = background/crack/scratch")
    ap.add_argument("--tag", default="",
                    help="suffix for the .onnx filename; REQUIRED to avoid "
                         "overwriting a differently-headed export of the same model")
    ap.add_argument("--resize", action="store_true",
                    help="the weights were trained on whole frames scaled to 256; "
                         "stamped so the app feeds them the same way")
    ap.add_argument("--size", type=int, default=SIZE,
                    help="input edge the weights were trained at; stamped into the "
                         "graph so the app feeds them the same scale")
    ap.add_argument("--prep", default=None,
                    help="input transform the weights were trained with, stamped into "
                         "the graph metadata so app/inference.py applies it without "
                         "being told")
    args = ap.parse_args()

    busy = machine_busy()
    if busy > 3 and not args.allow_busy:
        print(f"REFUSING: {busy} python processes running. Latency measured under "
              f"contention is meaningless (this project saw 96 / 103 / 412 ms for the "
              f"same model). Re-run when idle, or pass --allow-busy to override.")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        try:
            m = M.build(name, classes=args.classes)
            if args.weights and Path(args.weights).exists() and len(args.models) == 1:
                # A run's final `.pt` is a bare state dict; the `.best.pt` written on
                # every improvement wraps it under "state". Both are legitimate things
                # to export -- a run cut short by the clock only has the second.
                blob = torch.load(args.weights, map_location="cpu")
                m.load_state_dict(blob["state"] if isinstance(blob, dict)
                                  and "state" in blob else blob)
            m.eval()
            params = M.count_params(m)
            # The head is part of what the file IS, not metadata about it. A 3-class
            # export written over the binary one would leave every stored latency and
            # size figure describing a model that no longer sits at that path.
            stem = name.replace('/', '_') + (f"_{args.tag}" if args.tag else "")
            if args.classes != 1 and not args.tag:
                raise ValueError("--tag is required when --classes != 1, or this "
                                 "export would overwrite the binary one")
            onnx_path = OUT / f"{stem}.onnx"
            torch.onnx.export(m, torch.randn(1, 3, args.size, args.size), str(onnx_path),
                              input_names=["input"], output_names=["logits"],
                              opset_version=17,
                              dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}})
            stamp(onnx_path, prep=args.prep or "", classes=args.classes,
                  resize=int(args.resize), size=args.size)
            par = check_parity(m, onnx_path, args.size, args.classes)
            t_med, t_p90 = bench_torch_cpu(m, args.size)
            o_med, o_p90 = bench_onnx(onnx_path, args.size)
            mb = onnx_path.stat().st_size / 1e6
            rows.append(dict(model=name, classes=args.classes, size=args.size,
                             file=onnx_path.name,
                             params_m=params / 1e6, onnx_mb=mb,
                             torch_cpu_ms=t_med, torch_p90=t_p90,
                             onnx_1thread_ms=o_med, onnx_p90=o_p90, parity=par))
            print(f"{name:<40} {params/1e6:>7.3f}M  {mb:>6.1f}MB  "
                  f"torch {t_med:>6.1f}ms  onnx-1t {(o_med or float('nan')):>6.1f}ms")
            if par is None:
                print("    parity: NOT CHECKED (onnxruntime missing) -- this export is "
                      "unverified and must not be shipped as the measured model")
            else:
                ok = (par["max_abs_logit_diff"] < 1e-3
                      and par["argmax_agreement"] > 0.999)
                print(f"    parity: max|dlogit|={par['max_abs_logit_diff']:.2e} "
                      f"argmax_agree={par['argmax_agreement']:.5f} "
                      f"{'OK' if ok else '*** MISMATCH ***'}")
                if not ok:
                    print("    the exported graph does not reproduce the torch model. "
                          "Every metric in data/bench describes the torch weights, so "
                          "shipping this file would ship an unmeasured model.")
        except Exception as e:  # noqa: BLE001
            print(f"{name:<40} FAILED: {type(e).__name__}: {str(e)[:70]}")

    (OUT / "latency.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {OUT/'latency.json'}")
    print("NOTE: single-thread desktop CPU. A mid-range phone is ~3-6x slower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
