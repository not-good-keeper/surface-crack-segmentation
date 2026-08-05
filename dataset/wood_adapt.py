"""Wood adapter, one worker per shard.

data/wood_yield.csv already says which ids carry crack pixels, so the ~8,000 large BMPs
that do not are never decoded -- about 20x faster than the serial version in
adapters.py. Only `Crack` is foreground; knots, resin, marrow and blue stain are real
defects but not cracks, so those images become hard negatives.
"""
import argparse
import csv
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/clean"
MAX_SIDE = 1024
# Crack only. knot_with_crack outlines the whole knot, not the crack in it -- round
# blobs at 2.5x the area of a real crack. Images annotated only with it are excluded
# rather than made negative: they do contain a crack.
CRACK_BGR = [(100, 0, 255)]                 # #FF0064 Crack
KNOTCRACK_BGR = [(0, 175, 255)]             # #FFAF00 knot_with_crack -> exclude image
OTHER_BGR = [(0, 255, 0), (0, 0, 255), (100, 100, 0), (255, 0, 255),
             (255, 0, 0), (0, 100, 255), (255, 255, 16), (0, 64, 0)]


def imdec(buf, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)


def _shard(job):
    shard_path, crack_ids, hard_cap, clean_cap = job
    maps = zipfile.ZipFile(RAW / "Wood_Semantic_Maps.zip")
    map_by_id = {Path(n).stem.replace("_segm", ""): n
                 for n in maps.namelist() if n.lower().endswith(".bmp")}
    z = zipfile.ZipFile(shard_path)
    names = [n for n in z.namelist()
             if n.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]

    rows, n_hard, n_clean, n_knotonly = [], 0, 0, 0
    for n in names:
        sid = Path(n).stem
        is_crack = sid in crack_ids
        # Only crack images are always processed. Others are sampled up to a cap, so
        # we never pay to decode thousands of near-identical clean boards.
        if not is_crack and n_hard >= hard_cap and n_clean >= clean_cap:
            continue
        mn = map_by_id.get(sid)
        if mn is None:
            continue
        img = imdec(z.read(n))
        if img is None:
            continue
        smap = imdec(maps.read(mn))
        if smap is None:
            continue
        if smap.shape[:2] != img.shape[:2]:
            smap = cv2.resize(smap, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        sm = smap.astype(np.int16)
        crack = np.zeros(sm.shape[:2], bool)
        for c in CRACK_BGR:
            crack |= np.abs(sm - c).max(2) <= 30
        knotcrack = np.zeros(sm.shape[:2], bool)
        for c in KNOTCRACK_BGR:
            knotcrack |= np.abs(sm - c).max(2) <= 30
        other = np.zeros(sm.shape[:2], bool)
        for c in OTHER_BGR:
            other |= np.abs(sm - c).max(2) <= 30

        # unlocalisable crack: real crack, knot-shaped mask -> drop the image
        if knotcrack.any() and not crack.any():
            n_knotonly += 1
            continue

        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            crack = cv2.resize(crack.astype(np.uint8), (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            other = cv2.resize(other.astype(np.uint8), (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

        if crack.any():
            role, tier, mask = "positive", "A", crack.astype(np.uint8) * 255
        elif other.any():
            if n_hard >= hard_cap:
                continue
            role, tier, mask = "hard_negative", "B", None
            n_hard += 1
        else:
            if n_clean >= clean_cap:
                continue
            role, tier, mask = "negative", "C", None
            n_clean += 1

        name = f"wood__{sid}"
        cv2.imwrite(str(CLEAN / "images" / f"{name}.png"), img)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        cv2.imwrite(str(CLEAN / "masks" / f"{name}.png"), mask)
        rows.append(dict(name=name, source="wood", sid=sid, material="wood",
                         tier=tier, role=role, h=img.shape[0], w=img.shape[1],
                         crack_px=int((mask > 0).sum())))
    z.close()
    return Path(shard_path).name, rows, n_knotonly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-cap", type=int, default=250, help="per shard")
    ap.add_argument("--clean-cap", type=int, default=400, help="per shard")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    (CLEAN / "images").mkdir(parents=True, exist_ok=True)
    (CLEAN / "masks").mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(ROOT / "data/wood_yield.csv", dtype={"image_id": str})
    crack_ids = set(y.loc[y.any_crack, "image_id"])
    shards = sorted(RAW.glob("Wood_Images*.zip"))
    print(f"[wood] {len(shards)} shards, {len(crack_ids)} crack ids known upfront")

    jobs = [(str(s), crack_ids, args.hard_cap, args.clean_cap) for s in shards]
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        total_knot = 0
        for nm, rows, nk in ex.map(_shard, jobs):
            total_knot += nk
            pos = sum(1 for r in rows if r["role"] == "positive")
            print(f"  {nm}: {len(rows)} rows ({pos} crack)", flush=True)
            all_rows.extend(rows)

    # merge into the shared index
    out = CLEAN / "index_raw.csv"
    existing = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
        new = {r["name"] for r in all_rows}
        existing = [r for r in existing if r["name"] not in new]
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        wr.writeheader()
        wr.writerows(existing + all_rows)

    pos = sum(1 for r in all_rows if r["role"] == "positive")
    hard = sum(1 for r in all_rows if r["role"] == "hard_negative")
    clean = sum(1 for r in all_rows if r["role"] == "negative")
    print(f"\n[wood] {pos} crack / {hard} hard-neg / {clean} clean")
    print(f"[wood] index now {len(existing)+len(all_rows)} rows")

if __name__ == "__main__":
    main()
