"""Batch inspection: a directory of captures -> CSV report + JSON records + overlays.

    python app/batch.py --model data/export/<name>.onnx --images <dir> --out data/inspect

Same `Inspector` and same `Profile` as the interactive view, so a report generated here
is directly comparable with what an operator saw on screen. That is the point of the
file: not a second implementation, a second entry point.

Every processed image appends an audit line, which is what makes an inspection history
traceable without a server.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Inspector                       # noqa: E402
from postprocess import Profile, record_to_rows       # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/inspect")
    ap.add_argument("--station", default="LINE-1")
    ap.add_argument("--material")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    # Profile is not overridable from here. It was, and the two entry points drifted:
    # the CLI ran 0.50/0.50 while the app ran the values chosen on val, so a report and
    # the screen an operator approved disagreed about what counted as a defect.
    profile = Profile()
    insp = Inspector(args.model, profile, station_id=args.station)

    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        print(f"no images under {args.images}")
        return 1
    print(f"[batch] {len(paths)} images | model {insp.model_version} | "
          f"profile {profile.profile_id}")

    records, rows, audit, failed = [], [], [], 0
    for i, p in enumerate(paths, 1):
        raw = p.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # An unreadable capture is a distinct outcome. An empty successful record
            # here would enter the QC history as "inspected, no defect found".
            failed += 1
            audit.append(dict(path=str(p), status="acquisition_failure",
                              model_version=insp.model_version,
                              station_id=args.station,
                              processing_profile=profile.profile_id))
            print(f"  [{i}/{len(paths)}] {p.name}: UNREADABLE")
            continue

        rec, ov, _ = insp.inspect(img, raw, product_id=p.stem, material=args.material)
        dst = out / "overlays" / f"{p.stem}.png"
        if not cv2.imwrite(str(dst), ov):
            raise IOError(f"failed to write overlay {dst}")
        rec["overlay_path"] = str(dst)
        records.append(rec)
        rows += record_to_rows(rec)
        audit.append(dict(path=str(p), status="ok", n_regions=len(rec["regions"]),
                          **{k: rec[k] for k in ("image_sha256", "model_version",
                                                 "station_id", "processing_profile",
                                                 "processed_at")}))
        if i % 50 == 0 or i == len(paths):
            clean = sum(r["empty"] for r in records)
            print(f"  [{i}/{len(paths)}] {clean} clean, {len(records)-clean} with "
                  f"defects, {failed} failed", flush=True)

    pd.DataFrame(rows).to_csv(out / "report.csv", index=False)
    (out / "records.json").write_text(json.dumps(records, indent=1))
    with (out / "audit.jsonl").open("a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(a) + "\n" for a in audit)

    clean = sum(r["empty"] for r in records)
    print(f"\n[batch] {len(records)} inspected ({clean} clean), {failed} failed")
    print(f"[batch] -> {out/'report.csv'} | {out/'records.json'} | {out/'audit.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
