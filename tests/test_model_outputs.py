"""Unit tests for the model and the record it produces, on real held-out images.

    python tests/test_model_outputs.py [path/to/model.onnx]

app/test_pipeline.py tests post-processing against shapes whose geometry is known by
construction, and never runs a real image through a real model. This is the other half:
40 real, held-out photographs (tests/build_fixtures.py) through the exported ONNX, via
exactly the path app/app.py uses.

The assertions are deliberately of two kinds.

  Contract assertions are exact and must never fail: output resolution, schema, class
  vocabulary, determinism, geometry inside the image. These describe the interface, not
  the weights, so retraining cannot legitimately break them.

  Behavioural assertions are aggregate and have loose floors: detection rate over
  defective fixtures, marked area over clean ones. Per-image accuracy floors on 40
  images would be a brittle re-implementation of the benchmark, and bench/final_eval.py
  already measures accuracy properly on thousands. What belongs here is "the model is
  still doing its job at all" -- the regression that a metric table would show only
  after someone thought to look.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT / "app"))

FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILURES.append(msg)


def load_fixtures():
    meta = json.loads((FX / "fixtures.json").read_text())
    out = []
    for r in meta["images"]:
        p = FX / "images" / f"{r['name']}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out.append((r, img, p.read_bytes()))
    return meta, out


def main() -> int:
    if not (FX / "fixtures.json").exists():
        print("no fixtures -- run: python tests/build_fixtures.py")
        return 1
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("SKIP: onnxruntime not installed")
        return 0

    models = ([Path(sys.argv[1])] if len(sys.argv) > 1
              else sorted((ROOT / "data/export").glob("*.onnx"),
                          key=lambda p: p.stat().st_mtime, reverse=True))
    models = [m for m in models if m.exists()]
    if not models:
        print("SKIP: no exported model in data/export")
        return 0

    from inference import Inspector
    from postprocess import Profile

    meta, fixtures = load_fixtures()
    insp = Inspector(models[0], Profile())
    spec = insp.describe()
    print(f"model   {spec['file']}")
    print(f"contract {spec['input_size']} px | prep {spec['prep']} | "
          f"{spec['input_mode']} | {spec['classes']}-class")
    print(f"fixtures {len(fixtures)} real held-out images "
          f"({meta['n_with_defect']} defective, {meta['n_clean']} clean)\n")

    results, times = [], []
    for r, img, raw in fixtures:
        t0 = time.perf_counter()
        rec, ov, cmap = insp.inspect(img, raw, product_id=r["name"],
                                     material=r["material"])
        times.append((time.perf_counter() - t0) * 1000)
        results.append((r, img, rec, ov, cmap))

    # ---- contract: exact, and independent of the weights --------------------
    print("output contract")
    check(all(c.shape == i.shape[:2] for _, i, _, _, c in results),
          "class map is at the source resolution for every fixture")
    check(all(o.shape == i.shape for _, i, _, o, _ in results),
          "overlay is at the source resolution for every fixture")
    check(all(set(np.unique(c)) <= {0, 1, 2} for *_, c in results),
          "class map contains only background/crack/scratch")

    required = {"schema_version", "image_sha256", "model_version", "station_id",
                "processed_at", "processing_profile", "profile", "regions", "empty"}
    check(all(required <= set(rec) for _, _, rec, _, _ in results),
          "every record carries the full ARCHITECTURE 8.1 schema")
    check(all(rec["empty"] == (not rec["regions"]) for _, _, rec, _, _ in results),
          "`empty` agrees with the region list on every record")
    check(all(r["type"] in ("crack", "scratch")
              for _, _, rec, _, _ in results for r in rec["regions"]),
          "no region carries a type outside the taxonomy")

    print("\nregion geometry")
    bad_box = bad_num = 0
    for _, img, rec, _, _ in results:
        h, w = img.shape[:2]
        for g in rec["regions"]:
            x, y, bw, bh = g["bbox_xywh"]
            if x < 0 or y < 0 or x + bw > w or y + bh > h:
                bad_box += 1
            vals = (g["area_px"], g["length_px"], g["max_width_px"])
            if any(v is None or not np.isfinite(v) or v <= 0 for v in vals):
                bad_num += 1
    check(bad_box == 0, f"every bounding box lies inside its image ({bad_box} outside)")
    check(bad_num == 0, f"area, length and width are finite and positive ({bad_num} bad)")

    print("\ndeterminism (NFR-06)")
    r0, img0, rec0, _, _ = results[0]
    again, _, _ = insp.inspect(img0, (FX / "images" / f"{r0['name']}.png").read_bytes())
    key = lambda rc: [(g["type"], g["area_px"], g["length_px"], g["max_width_px"])  # noqa: E731
                      for g in rc["regions"]]
    check(key(rec0) == key(again), "the same image yields identical regions on re-run")

    # ---- behaviour: aggregate, with loose floors ----------------------------
    print("\nbehaviour on real images")
    defective = [(r, c) for r, _, _, _, c in
                 [(r, i, rec, o, c) for r, i, rec, o, c in results] if r["has_defect"]]
    clean = [(r, c) for r, _, _, _, c in
             [(r, i, rec, o, c) for r, i, rec, o, c in results] if not r["has_defect"]]

    det = sum((c > 0).any() for _, c in defective) / max(len(defective), 1)
    check(det >= 0.70,
          f"detects something on {det:.0%} of defective fixtures (floor 70 %)")

    marked = [float((c > 0).mean()) for _, c in clean]
    worst = max(marked) if marked else 0.0
    check(worst <= 0.05,
          f"marks at most {worst:.2%} of any clean image as defect (ceiling 5 %)")
    check(float(np.mean(marked or [0])) <= 0.02,
          f"marks {float(np.mean(marked or [0])):.2%} of clean pixels on average "
          f"(ceiling 2 %)")

    # A model that fires everywhere would pass detection and fail the product.
    over = [float((c > 0).mean()) for _, c in defective]
    check(max(over or [0]) <= 0.60,
          f"marks at most {max(over or [0]):.0%} of any defective image (ceiling 60 %)")

    print("\nlatency (indicative -- CPU, machine may be busy)")
    print(f"  median {np.median(times):.0f} ms | p90 {np.percentile(times, 90):.0f} ms "
          f"| max {max(times):.0f} ms")

    print(f"\n{'ALL PASSED' if not FAILURES else str(len(FAILURES)) + ' FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
