"""Kaggle downloader using a KGAT_ access token.

The bundled kaggle CLI (1.7.x) only understands kaggle.json / KAGGLE_USERNAME +
KAGGLE_KEY and rejects the newer KGAT_ access-token format outright. That token
does however work as a plain bearer credential against the REST API -- verified:
  competitions/data/list/severstal-steel-defect-detection
  -> 401 without the header, 200 with it.
So we skip the CLI and talk to the API directly.

Usage:
    python dataset/kaggle_fetch.py --list
    python dataset/kaggle_fetch.py --only neu_seg severstal
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(__file__).resolve().parent / "sources.yaml"
RAW = ROOT / "data/raw"
API = "https://www.kaggle.com/api/v1"
CHUNK = 1 << 20


def token():
    for p in (Path.home() / ".kaggle/access_token", Path.home() / ".kaggle/kaggle.json"):
        if p.exists():
            txt = p.read_text().strip()
            if txt.startswith("{"):
                return json.loads(txt).get("key", "")
            return txt
    if t := os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return t
    raise SystemExit("no Kaggle credential found (~/.kaggle/access_token)")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def url_for(spec):
    ref = spec["ref"]
    if spec["kind"] == "competition":
        return f"{API}/competitions/data/download-all/{ref}"
    return f"{API}/datasets/download/{ref}"


def fetch(name, spec, hdr):
    dest = RAW / f"kaggle_{name}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{name}] cached {human(dest.stat().st_size)} -> {dest.name}")
        return True

    part = dest.with_suffix(".zip.part")
    have = part.stat().st_size if part.exists() else 0
    h = dict(hdr)
    if have:
        h["Range"] = f"bytes={have}-"

    try:
        with requests.get(url_for(spec), headers=h, stream=True,
                          timeout=120, allow_redirects=True) as r:
            if r.status_code == 401:
                print(f"[{name}] 401 UNAUTHORIZED - token rejected")
                return False
            if r.status_code == 403:
                print(f"[{name}] 403 FORBIDDEN - this is a CONSENT gate, not a bad token.")
                print(f"[{name}]   Accept the rules at kaggle.com/c/{spec['ref']}/rules "
                      f"(or the dataset's terms), then re-run.")
                return False
            if r.status_code == 404:
                print(f"[{name}] 404 - ref '{spec['ref']}' not found; check the slug")
                return False
            if have and r.status_code == 200:
                have = 0
                part.unlink(missing_ok=True)
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0)) + have
            RAW.mkdir(parents=True, exist_ok=True)
            t0, last = time.time(), have
            with part.open("ab" if have else "wb") as fh:
                for chunk in r.iter_content(CHUNK):
                    fh.write(chunk)
                    have += len(chunk)
                    if time.time() - t0 > 20:
                        rate = (have - last) / (time.time() - t0)
                        pct = f"{100*have/total:.0f}%" if total else "?"
                        print(f"[{name}] {pct} {human(have)} {human(rate)}/s", flush=True)
                        t0, last = time.time(), have
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAILED: {e}")
        return False

    size = part.stat().st_size
    if size < 4096:
        print(f"[{name}] FAILED: suspiciously small ({size} B) - likely an error page")
        print(f"[{name}] body: {part.read_bytes()[:200]!r}")
        return False

    part.replace(dest)
    sha = hashlib.sha256()
    with dest.open("rb") as fh:
        while c := fh.read(1 << 22):
            sha.update(c)
    lock_path = RAW / ".lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    lock[f"kaggle_{name}"] = {
        "ref": spec["ref"], "kind": spec["kind"], "bytes": size,
        "sha256": sha.hexdigest(), "tier": spec.get("tier"),
        "license": spec.get("license"), "materials": spec.get("materials"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))
    print(f"[{name}] DONE {human(size)} sha256={sha.hexdigest()[:16]}... -> {dest.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    todo = {n: s for n, s in cfg.get("kaggle", {}).items()
            if s.get("enabled", True) and (not args.only or n in args.only)}

    if args.list:
        for n, s in cfg.get("kaggle", {}).items():
            print(f"{n:<20}{s['kind']:<13}{str(s.get('tier')):<4}{s['ref']}")
        return 0

    hdr = {"Authorization": f"Bearer {token()}",
           "User-Agent": "tinyseg-dataset/1.0"}
    print(f"fetching {len(todo)} kaggle source(s)")
    results = {n: fetch(n, s, hdr) for n, s in todo.items()}

    bad = [n for n, v in results.items() if not v]
    print(f"\n=== {len(results)-len(bad)} ok, {len(bad)} failed ===")
    for n in bad:
        print(f"  FAILED: {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
