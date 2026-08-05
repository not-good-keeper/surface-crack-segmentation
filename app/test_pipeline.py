"""Checks on the inspection path that do not need a trained model.

    python app/test_pipeline.py

These exist because the app and the batch CLI are two entry points to one pipeline, and
the failure that matters is them silently disagreeing -- an operator approving a part on
screen while the QC report records something else. Geometry is asserted against shapes
whose true measurements are known by construction, not against whatever the code
currently returns.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import (Profile, build_record, class_map, extract_regions,  # noqa: E402
                         overlay, record_to_rows, softmax)

FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILURES.append(msg)


def logits_from(cmap, n_cls=3, conf=8.0):
    lg = np.zeros((n_cls, *cmap.shape), np.float32)
    for c in range(n_cls):
        lg[c][cmap == c] = conf
    return lg


def test_geometry_is_correct_by_construction():
    print("\ngeometry on shapes with known measurements")
    cmap = np.zeros((80, 80), np.int32)
    # A horizontal bar: 40 long, 3 wide. Length and max width are known exactly.
    cmap[10:13, 20:60] = 1
    prof = Profile(min_area_px=5, min_skeleton_px=3)
    regs = extract_regions(cmap, prof)
    check(len(regs) == 1, f"one region found (got {len(regs)})")
    r = regs[0]
    check(r['area_px'] == 120, f"area is 3x40=120 (got {r['area_px']})")
    check(abs(r['length_px'] - 39) <= 2, f"length ~39 px (got {r['length_px']})")
    check(abs(r['max_width_px'] - 3) <= 1.0, f"max width ~3 px (got {r['max_width_px']})")

    # Locally wide defect: area/length would report the average and hide the bulge.
    # The bulge is square on purpose. Max width is twice the largest inscribed-circle
    # radius, so for a section that is tall but narrow the answer is set by the NARROW
    # dimension -- correct for the metric, but it would make this test assert a number
    # that depends on the bulge's aspect ratio rather than on the property being tested.
    cmap2 = np.zeros((80, 80), np.int32)
    cmap2[40:42, 10:70] = 1
    cmap2[35:47, 35:47] = 1                    # a 12x12 bulge in the middle
    r2 = extract_regions(cmap2, prof)[0]
    naive = r2['area_px'] / max(r2['length_px'], 1)
    check(r2['max_width_px'] >= 10,
          f"max width catches the bulge (got {r2['max_width_px']}, "
          f"area/length would say {naive:.1f})")


def test_classes_are_mutually_exclusive():
    print("\nclass decision")
    prob = np.zeros((3, 8, 8), np.float32)
    prob[1, 2, 2], prob[2, 2, 2] = 0.55, 0.40    # both over floor, crack wins
    prob[1, 4, 4], prob[2, 4, 4] = 0.30, 0.70    # scratch wins
    prob[1, 6, 6], prob[2, 6, 6] = 0.20, 0.20    # neither clears the floor
    cm = class_map(prob, Profile(crack_thresh=0.5, scratch_thresh=0.5))
    check(cm[2, 2] == 1, "higher-scoring class wins when both pass")
    check(cm[4, 4] == 2, "scratch wins when it scores higher")
    check(cm[6, 6] == 0, "below both floors is background")
    check(set(np.unique(cm)) <= {0, 1, 2}, "no pixel carries two classes")


def test_min_area_filter():
    print("\nnoise filtering")
    cmap = np.zeros((60, 60), np.int32)
    cmap[10:13, 10:40] = 1                       # keep: 90 px
    cmap[50, 50] = 1                             # drop: 1 px speck
    regs = extract_regions(cmap, Profile(min_area_px=24, min_skeleton_px=3))
    check(len(regs) == 1, f"speck below min_area is dropped (got {len(regs)})")


def test_touching_classes_stay_separate():
    print("\nadjacent defects of different types")
    cmap = np.zeros((60, 60), np.int32)
    cmap[20:23, 5:30] = 1
    cmap[20:23, 30:55] = 2                       # touching, different class
    regs = extract_regions(cmap, Profile(min_area_px=10, min_skeleton_px=3))
    check(len(regs) == 2, f"two regions, not one merged blob (got {len(regs)})")
    check({r['type'] for r in regs} == {"crack", "scratch"}, "both types reported")


def test_clean_product_is_explicit():
    print("\nclean vs missing")
    rec = build_record(b"bytes", [], Profile(), "m")
    check(rec["empty"] is True, "clean product sets empty=True")
    rows = record_to_rows(rec)
    check(len(rows) == 1 and rows[0]["type"] == "none",
          "a clean product still emits one CSV row")


def test_determinism():
    print("\ndeterminism (NFR-6)")
    rng = np.random.default_rng(0)
    prob = softmax(rng.normal(0, 3, (3, 64, 64)).astype(np.float32))
    prof = Profile()
    a = [(r['id'], r['type'], r['area_px'], r['length_px'], r['max_width_px'])
         for r in extract_regions(class_map(prob, prof), prof)]
    b = [(r['id'], r['type'], r['area_px'], r['length_px'], r['max_width_px'])
         for r in extract_regions(class_map(prob, prof), prof)]
    check(a == b, "identical input yields identical regions")


def test_overlay_survives_rescaling():
    print("\noverlay in source coordinates")
    cmap = np.zeros((256, 256), np.int32)
    cmap[100:104, 50:200] = 1
    regs = extract_regions(cmap, Profile())
    big = np.full((720, 1280, 3), 180, np.uint8)
    ov = overlay(big, cmap, regs)
    check(ov.shape == big.shape,
          f"overlay keeps the source resolution (got {ov.shape})")
    check(bool((ov != big).any()), "overlay actually marks something")


def test_app_and_batch_agree():
    """Verification item 7: one image, both entry points, identical geometry."""
    print("\napp and batch produce identical geometry")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("  SKIP  onnxruntime not installed")
        return
    models = sorted((Path(__file__).resolve().parent.parent
                     / "data/export").glob("*.onnx"))
    if not models:
        print("  SKIP  no exported model yet")
        return
    from inference import Inspector

    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    raw = buf.tobytes()
    prof = Profile()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "part.png"
        p.write_bytes(raw)
        # the app path: bytes decoded in memory
        a_rec, _, _ = Inspector(models[0], prof).inspect(
            cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), raw)
        # the batch path: same bytes read from disk
        b_raw = p.read_bytes()
        b_rec, _, _ = Inspector(models[0], prof).inspect(
            cv2.imdecode(np.frombuffer(b_raw, np.uint8), cv2.IMREAD_COLOR), b_raw)

    key = lambda r: [(x["type"], x["area_px"], x["length_px"], x["max_width_px"])  # noqa: E731
                     for x in r["regions"]]
    check(key(a_rec) == key(b_rec), "same image -> same regions on both paths")
    check(a_rec["image_sha256"] == b_rec["image_sha256"], "same image hash")
    check(a_rec["processing_profile"] == b_rec["processing_profile"],
          "same processing profile recorded")


if __name__ == "__main__":
    for t in (test_geometry_is_correct_by_construction,
              test_classes_are_mutually_exclusive,
              test_min_area_filter,
              test_touching_classes_stay_separate,
              test_clean_product_is_explicit,
              test_determinism,
              test_overlay_survives_rescaling,
              test_app_and_batch_agree):
        t()
    print(f"\n{'ALL PASSED' if not FAILURES else str(len(FAILURES)) + ' FAILURE(S)'}")
    sys.exit(1 if FAILURES else 0)
