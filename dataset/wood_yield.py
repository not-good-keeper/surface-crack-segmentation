"""Decide which wood image shards are worth downloading.

The Kodytek wood corpus is 10 shards x ~15 GB = ~155 GB, but the semantic maps for
ALL 20,276 images are a single 207 MB download. So:

  1. Scan every mask locally and find which image IDs actually contain crack pixels.
  2. Learn each shard's file list by range-reading only its zip central directory
     (a few hundred KB out of 15 GB each) -- no full download required.
  3. Rank shards by crack yield and enable only the best ones.

Colour codes come from Semantic Map Specification.txt:
    Crack            = #FF0064
    knot_with_crack  = #FFAF00
Everything else (live/dead knots, resin, marrow, quartzity, blue stain, overgrown)
is NOT a crack and becomes a Tier-B hard negative later.
"""
import argparse
import io
import re
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
MAPS_ZIP = ROOT / "data/raw/Wood_Semantic_Maps.zip"
CFG = Path(__file__).resolve().parent / "sources.yaml"
OUT = ROOT / "data/wood_yield.csv"

# BGR, because cv2 decodes to BGR
CRACK_BGR = (100, 0, 255)      # FF0064
KNOTCRACK_BGR = (0, 175, 255)  # FFAF00
TOL = 30                       # BMPs are palettised/exact, but stay tolerant


def _count(args):
    """Count crack-class pixels in one mask. Runs in a worker process."""
    zpath, names = args
    out = []
    with zipfile.ZipFile(zpath) as z:
        for name in names:
            try:
                buf = np.frombuffer(z.read(name), np.uint8)
                m = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if m is None:
                    continue
                crack = (np.abs(m.astype(np.int16) - CRACK_BGR).max(2) <= TOL).sum()
                knot = (np.abs(m.astype(np.int16) - KNOTCRACK_BGR).max(2) <= TOL).sum()
                img_id = Path(name).stem.replace("_segm", "")
                out.append((img_id, int(crack), int(knot), m.shape[0], m.shape[1]))
            except Exception:  # noqa: BLE001 - one bad mask must not kill the scan
                continue
    return out


def scan_masks(workers=8) -> pd.DataFrame:
    if OUT.exists():
        print(f"[scan] reusing {OUT}")
        return pd.read_csv(OUT, dtype={"image_id": str})

    with zipfile.ZipFile(MAPS_ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".bmp")]
    print(f"[scan] {len(names)} masks, {workers} workers")

    chunks = [(str(MAPS_ZIP), names[i::workers * 4]) for i in range(workers * 4)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_count, chunks), 1):
            rows.extend(part)
            print(f"[scan] chunk {i}/{len(chunks)}  rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["image_id", "crack_px", "knot_crack_px", "h", "w"])
    df["any_crack"] = (df.crack_px > 0) | (df.knot_crack_px > 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[scan] wrote {OUT}")
    return df


# ---------------------------------------------------------------- remote zip listing
def _range(url, start, end) -> bytes:
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
    r.raise_for_status()
    return r.content


def _size(url):
    r = requests.get(url, headers={"Range": "bytes=0-1"}, timeout=90, stream=True)
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    r.close()
    m = re.search(r"/(\d+)$", cr)
    if not m:
        raise RuntimeError(f"no Content-Range from {url}")
    return int(m.group(1))


def remote_namelist(url):
    """Read a remote zip's central directory without downloading the archive.

    These shards are >4 GB so they are always ZIP64: locate the EOCD, follow the
    ZIP64 locator to the ZIP64 EOCD, then range-fetch just the central directory.
    """
    total = _size(url)
    tail = _range(url, max(0, total - 65_536), total - 1)

    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD found")
    cd_size = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cd_off = struct.unpack("<I", tail[i + 16:i + 20])[0]

    j = tail.rfind(b"PK\x06\x07")  # ZIP64 end-of-central-directory locator
    if j >= 0 or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        z64 = _range(url, z64_off, z64_off + 55)
        if z64[:4] != b"PK\x06\x06":
            raise RuntimeError("bad ZIP64 EOCD")
        cd_size = struct.unpack("<Q", z64[40:48])[0]
        cd_off = struct.unpack("<Q", z64[48:56])[0]

    cd = _range(url, cd_off, cd_off + cd_size - 1)

    names, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        n_len, x_len, c_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        names.append(cd[p + 46:p + 46 + n_len].decode("utf-8", "replace"))
        p += 46 + n_len + x_len + c_len
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top", type=int, default=3, help="how many shards to enable")
    ap.add_argument("--apply", action="store_true", help="write enabled:true into sources.yaml")
    args = ap.parse_args()

    df = scan_masks(args.workers)
    crack_ids = set(df.loc[df.any_crack, "image_id"])
    print(f"\n[scan] {len(df)} masks | {len(crack_ids)} contain crack or knot_with_crack "
          f"({100*len(crack_ids)/max(len(df),1):.1f}%)")
    print(f"[scan] pure Crack class: {(df.crack_px>0).sum()} | "
          f"knot_with_crack: {(df.knot_crack_px>0).sum()}")

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    shards = {n: s for n, s in cfg["sources"].items() if n.startswith("wood_images")}

    print(f"\n[shards] reading central directories of {len(shards)} shards remotely ...")
    rank = []
    for name, spec in sorted(shards.items(), key=lambda kv: kv[1]["shard"]):
        try:
            names = remote_namelist(spec["url"])
            ids = {Path(n).stem for n in names if n.lower().endswith((".bmp", ".png", ".jpg"))}
            hits = len(ids & crack_ids)
            gb = spec["bytes"] / 1e9
            rank.append((name, spec["shard"], len(ids), hits, gb, hits / gb if gb else 0))
            print(f"  shard {spec['shard']:>2}: {len(ids):>5} images, "
                  f"{hits:>5} with cracks, {gb:.1f} GB -> {hits/gb:.0f} cracks/GB")
        except Exception as e:  # noqa: BLE001
            print(f"  shard {spec['shard']:>2}: FAILED ({e})")

    if not rank:
        print("no shard listings succeeded")
        return 1

    rank.sort(key=lambda r: -r[5])
    print(f"\n[rank] best value shards: {[r[1] for r in rank[:args.top]]}")
    chosen = [r[0] for r in rank[:args.top]]
    total_gb = sum(r[4] for r in rank[:args.top])
    total_hits = sum(r[3] for r in rank[:args.top])
    print(f"[rank] enabling {chosen} = {total_gb:.1f} GB for {total_hits} crack images")

    if args.apply:
        txt = CFG.read_text(encoding="utf-8")
        for name in chosen:
            txt = re.sub(rf"(\b{name}:\s*\{{[^}}]*?)enabled: false", r"\1enabled: true", txt)
        CFG.write_text(txt, encoding="utf-8")
        print(f"[apply] sources.yaml updated -> run: python dataset/fetch.py")
    else:
        print("[dry-run] pass --apply to enable them in sources.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
