"""Fetch the Roboflow Universe sources listed in sources.yaml.

    python dataset/fetch_roboflow.py --list
    python dataset/fetch_roboflow.py --only rf_pipe_crack

Key comes from ~/.roboflow/api_key or ROBOFLOW_API_KEY; never written to the repo or
the lock file.

Export format is negotiated, not assumed -- a project may publish masks, COCO polygons
or boxes only. Mask formats are tried first and what actually arrived is recorded, so
a boxes-only project is visible instead of becoming a silent empty-mask source.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(__file__).resolve().parent / "sources.yaml"
RAW = ROOT / "data/raw/roboflow"

# Ordered by how directly usable the result is for pixel segmentation.
FORMATS = ["png-mask-semantic", "coco-segmentation", "coco"]


def api_key() -> str:
    p = Path.home() / ".roboflow/api_key"
    if p.exists():
        k = p.read_text().strip()
        if k:
            return k
    if k := os.environ.get("ROBOFLOW_API_KEY"):
        return k
    raise SystemExit("no Roboflow credential (~/.roboflow/api_key or "
                     "ROBOFLOW_API_KEY)")


def fetch_one(name: str, spec: dict, rf) -> dict:
    ws, _, proj = spec["ref"].partition("/")
    dest = RAW / name
    rec = dict(name=name, ref=spec["ref"], material=spec.get("material"),
               use=spec.get("use", []), mask_style=spec.get("mask_style"))

    if (dest / ".done.json").exists():
        got = json.loads((dest / ".done.json").read_text())
        print(f"[{name}] cached ({got.get('format')}, v{got.get('version')})")
        return got

    # Version discovery goes through the REST endpoint rather than the SDK helper.
    # A Universe project is browsable without having any GENERATED version, and the
    # SDK reports that case the same way it reports a transport failure, which would
    # make "this dataset cannot be downloaded at all" look like a flaky network.
    import requests
    try:
        d = requests.get(f"https://api.roboflow.com/{ws}/{proj}",
                         params={"api_key": rf.api_key}, timeout=60).json()
        ptype = d.get("project", {}).get("type", "?")
        rec["project_type"] = ptype
        vs = [int(v["id"].split("/")[-1]) for v in d.get("versions", []) if v.get("id")]
        if not vs:
            rec.update(status="no generated version -- browsable but not downloadable")
            print(f"[{name}] SKIP: no generated version ({ptype}, "
                  f"{d.get('project', {}).get('images', 0)} images)")
            return rec
        vnum = max(vs)
        version = rf.workspace(ws).project(proj).version(vnum)
        rec["version"] = vnum
    except Exception as e:  # noqa: BLE001
        rec.update(status=f"{type(e).__name__}: {str(e)[:110]}")
        print(f"[{name}] FAILED to resolve: {rec['status']}")
        return rec

    for fmt in FORMATS:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            version.download(fmt, location=str(dest), overwrite=True)
            n_img = sum(1 for _ in dest.rglob("*.jpg")) + \
                sum(1 for _ in dest.rglob("*.png"))
            rec.update(format=fmt, status="ok", n_files=n_img,
                       fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            (dest / ".done.json").write_text(json.dumps(rec, indent=1))
            flag = "" if fmt.startswith("png-mask") else "  <- NOT pixel masks"
            print(f"[{name}] v{vnum} {fmt}: {n_img} files{flag}")
            return rec
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:90]}"
            continue

    rec.update(status=f"no usable format ({last})")
    print(f"[{name}] FAILED: {rec['status']}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")).get("roboflow", {})
    todo = {n: s for n, s in cfg.items() if not args.only or n in args.only}

    if args.list:
        for n, s in todo.items():
            print(f"{n:<28} {s['ref']:<52} {s['material']:<8} "
                  f"use={s.get('use')} style={s.get('mask_style')}")
        return 0

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key())
    RAW.mkdir(parents=True, exist_ok=True)

    results = [fetch_one(n, s, rf) for n, s in todo.items()]
    ok = [r for r in results if r.get("status") == "ok"]
    masks = [r for r in ok if str(r.get("format", "")).startswith("png-mask")]
    print(f"\n=== {len(ok)}/{len(results)} fetched, {len(masks)} as pixel masks ===")
    for r in results:
        if r.get("status") != "ok":
            print(f"  MISSING {r['name']}: {r.get('status')}")
    (RAW / "manifest.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
