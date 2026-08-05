"""Per-source adapters -> one flat layout under data/clean/.

    images/{source}__{id}.png
    masks/{source}__{id}.png    binary {0,255}, PNG (never JPEG)
    index_raw.csv               source, material, tier, role

Roles: positive (real defect, foreground), hard_negative (defect-like but not ours),
negative (clean surface). docs/DATASET.md covers what each source contributes.
"""
import argparse
import csv
import io
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/clean"
IMG_DIR = CLEAN / "images"
MSK_DIR = CLEAN / "masks"
BIN_THRESH = 127  # anti-aliased edges are as wide as the crack core

# CrackSeg9k filename prefix -> material. Derived from the actual prefixes in
# the archive; 'a_/b_/c_/d_' are the masonry subset (verified visually: brick).
CS9K_MATERIAL = [
    ("noncrack", "concrete"), ("CRACK500", "asphalt"), ("GAPS", "asphalt"),
    ("cracktree", "asphalt"), ("CFD", "asphalt"), ("DeepCrack", "concrete"),
    ("Rissbilder", "concrete"), ("Volker", "concrete"), ("Ceramic", "ceramic"),
    ("Masonry", "masonry"),
]


def material_of(stem: str) -> str:
    for pref, mat in CS9K_MATERIAL:
        if stem.startswith(pref):
            return mat
    if stem[:2] in ("a_", "b_", "c_", "d_"):
        return "masonry"
    return "unknown"


def write_pair(rows, source, sid, material, tier, role, img, mask):
    """Persist one image (+optional mask) and record its index row."""
    if img is None or img.size == 0:
        return
    name = f"{source}__{sid}"
    cv2.imwrite(str(IMG_DIR / f"{name}.png"), img)
    if mask is None:
        mask = np.zeros(img.shape[:2], np.uint8)
    else:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask = ((mask > BIN_THRESH) * 255).astype(np.uint8)
    cv2.imwrite(str(MSK_DIR / f"{name}.png"), mask)
    rows.append(dict(name=name, source=source, sid=sid, material=material,
                     tier=tier, role=role, h=img.shape[0], w=img.shape[1],
                     crack_px=int((mask > 0).sum())))


def imdec(buf, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)


# ------------------------------------------------------------------ CrackSeg9k
def adapt_crackseg9k(rows, limit=None):
    v1 = RAW / "Final-Dataset-Vol1.zip"
    v2 = RAW / "Final-Dataset-Vol2.zip"
    if not v1.exists():
        print("[crackseg9k] SKIP (not downloaded)")
        return
    z1, z2 = zipfile.ZipFile(v1), zipfile.ZipFile(v2)

    # Final_Masks/Masks are the labels; Final_Masks/Heads is a saliency map.
    masks = {Path(n).stem: n for n in z1.namelist()
             if "Final_Masks/Masks/" in n and n.endswith(".png")}
    imgs = {Path(n).stem: ("1", n) for n in z1.namelist() if "/Images/" in n}
    imgs.update({Path(n).stem: ("2", n) for n in z2.namelist() if "/Images-2/" in n})

    stems = sorted(set(masks) & set(imgs))
    if limit:
        stems = stems[:limit]
    print(f"[crackseg9k] {len(stems)} paired image/mask (ignoring {len(masks)-len(stems)} unpaired)")

    for i, stem in enumerate(stems):
        vol, ipath = imgs[stem]
        img = imdec((z1 if vol == "1" else z2).read(ipath))
        msk = imdec(z1.read(masks[stem]), cv2.IMREAD_GRAYSCALE)
        mat = material_of(stem)
        role = "negative" if stem.startswith("noncrack") else "positive"
        write_pair(rows, "crackseg9k", stem, mat, "A" if role == "positive" else "C",
                   role, img, msk)
        if i % 2000 == 0:
            print(f"  ... {i}/{len(stems)}", flush=True)


# ------------------------------------------------------------------ Magnetic Tile
def adapt_magnetic_tile(rows, limit=None):
    p = RAW / "MagneticTile.zip"
    if not p.exists():
        print("[magnetic_tile] SKIP")
        return
    z = zipfile.ZipFile(p)
    names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png", ".bmp"))]
    # layout: MT_<Defect>/Imgs/xxx.jpg  +  xxx.png (mask alongside)
    imgs = [n for n in names if n.lower().endswith(".jpg")]
    masks = {Path(n).stem: n for n in names if n.lower().endswith(".png")}
    n_pos = n_neg = n_hard = 0
    for n in (imgs[:limit] if limit else imgs):
        cls = n.split("/")[1] if len(n.split("/")) > 2 else ""
        stem = Path(n).stem
        img = imdec(z.read(n))
        mk = masks.get(stem)
        msk = imdec(z.read(mk), cv2.IMREAD_GRAYSCALE) if mk else None
        if "Crack" in cls:
            role, tier = "positive", "A"; n_pos += 1
        elif "Free" in cls:
            role, tier, msk = "negative", "C", None; n_neg += 1
        else:  # Blowhole, Break, Fray, Uneven -> defects, but NOT cracks
            role, tier, msk = "hard_negative", "B", None; n_hard += 1
        write_pair(rows, "magnetic_tile", f"{cls}_{stem}", "ceramic", tier, role, img, msk)
    print(f"[magnetic_tile] {n_pos} crack / {n_hard} hard-neg / {n_neg} clean")


# ------------------------------------------------------------------ NEU (steel)
def adapt_neu(rows, limit=None):
    """NEU-DET steel: boxes only, no masks (the 'neu-seg' Kaggle ref is mislabelled).

    Crazing is dropped -- a real crack network with no mask is unusable as a positive
    and harmful as a negative. The other five classes are not cracks, so they make
    good hard negatives.
    """
    p = RAW / "kaggle_neu_seg.zip"
    if not p.exists():
        print("[neu_det] SKIP")
        return
    z = zipfile.ZipFile(p)
    names = [n for n in z.namelist()
             if "/images/" in n and n.lower().endswith((".jpg", ".png", ".bmp"))]
    n_hard = n_skip = 0
    for n in (names[:limit] if limit else names):
        cls = Path(n).stem.rsplit("_", 1)[0]
        if cls == "crazing":
            n_skip += 1
            continue
        img = imdec(z.read(n))
        if img is None:
            continue
        write_pair(rows, "neu_det", Path(n).stem, "steel", "B", "hard_negative", img, None)
        n_hard += 1
    print(f"[neu_det] {n_hard} steel hard-negatives "
          f"(scratches/inclusion/patches/pitted/rolled-in scale)")
    print(f"[neu_det] EXCLUDED {n_skip} 'crazing' images: real crack networks with no "
          f"masks -- unusable as positives, harmful as negatives")


# ------------------------------------------------------------------ negatives / backgrounds
def _flat_images(zpath, source, material, role, tier, rows, limit, exclude=()):
    if not zpath.exists():
        print(f"[{source}] SKIP")
        return
    opener = tarfile.open if zpath.suffix in (".gz", ".tar") else zipfile.ZipFile
    n = 0
    if opener is tarfile.open:
        with tarfile.open(zpath) as t:
            for m in t:
                if not m.isfile() or not m.name.lower().endswith((".jpg", ".png", ".bmp")):
                    continue
                if any(e in m.name.lower() for e in exclude):
                    continue
                img = imdec(t.extractfile(m).read())
                if img is None:
                    continue
                write_pair(rows, source, Path(m.name).stem, material, tier, role, img, None)
                n += 1
                if limit and n >= limit:
                    break
    else:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.lower().endswith((".jpg", ".png", ".bmp")):
                    continue
                if any(e in name.lower() for e in exclude):
                    continue
                img = imdec(z.read(name))
                if img is None:
                    continue
                write_pair(rows, source, Path(name).stem, material, tier, role, img, None)
                n += 1
                if limit and n >= limit:
                    break
    print(f"[{source}] {n} images ({role})")


# ------------------------------------------------------------------ KolektorSDD2
def adapt_kolektor(rows, limit=None):
    """Flat train/ and test/, XXXXX.png + XXXXX_GT.png.

    Defective images are excluded: the labels mix scratches, spots and real cracks
    with nothing to separate them. The clean surfaces are what we want anyway.
    """
    p = RAW / "KolektorSDD2.zip"
    if not p.exists():
        print("[kolektor] SKIP")
        return
    z = zipfile.ZipFile(p)
    pairs = {}
    for n in z.namelist():
        if not n.endswith(".png"):
            continue
        stem = Path(n).stem
        if stem.endswith("_GT"):
            pairs.setdefault(stem[:-3], {})["gt"] = n
        else:
            pairs.setdefault(stem, {})["img"] = n

    n_clean = n_skip = 0
    for stem, d in (list(pairs.items())[:limit] if limit else pairs.items()):
        if "img" not in d or "gt" not in d:
            continue
        gt = imdec(z.read(d["gt"]), cv2.IMREAD_GRAYSCALE)
        if gt is not None and (gt > 127).any():
            n_skip += 1           # defective: ambiguous crack/not-crack -> drop
            continue
        img = imdec(z.read(d["img"]))
        if img is None:
            continue
        write_pair(rows, "kolektor", stem, "plastic", "C", "negative", img, None)
        n_clean += 1
    print(f"[kolektor] {n_clean} clean metal/plastic backgrounds | "
          f"EXCLUDED {n_skip} defective (crack/scratch labels not separable)")


# ------------------------------------------------------------------ GC10-DET
def adapt_gc10(rows, limit=None):
    """10 metal defect classes, none of them cracks -- real crack-like clutter."""
    p = RAW / "kaggle_gc10det.zip"
    if not p.exists():
        print("[gc10] SKIP")
        return
    z = zipfile.ZipFile(p)
    names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))]
    n = 0
    for name in (names[:limit] if limit else names):
        img = imdec(z.read(name))
        if img is None:
            continue
        cls = name.split("/")[0]
        write_pair(rows, "gc10", f"c{cls}_{Path(name).stem}", "metal", "B",
                   "hard_negative", img, None)
        n += 1
    print(f"[gc10] {n} metal hard-negatives")


# ------------------------------------------------------------------ SDNET2018
def adapt_sdnet(rows, limit=None):
    limit = limit or 6000
    """Classification only. Non-cracked gives a large clean concrete pool; cracked
    is dropped for the same reason as NEU crazing -- real cracks, no masks.
    """
    p = RAW / "kaggle_sdnet2018.zip"
    if not p.exists():
        print("[sdnet] SKIP")
        return
    z = zipfile.ZipFile(p)
    clean = [n for n in z.namelist()
             if "non-cracked" in n.lower() and n.lower().endswith((".jpg", ".png"))]
    cracked = [n for n in z.namelist()
               if n.lower().endswith((".jpg", ".png")) and "non-cracked" not in n.lower()
               and "cracked" in n.lower()]
    step = max(1, len(clean) // limit)      # even spread over Pavements/Walls/Decks
    taken = clean[::step][:limit]
    for name in taken:
        img = imdec(z.read(name))
        if img is None:
            continue
        sub = name.split("/")[0]
        write_pair(rows, "sdnet", f"{sub}_{Path(name).stem}", "concrete", "C",
                   "negative", img, None)
    print(f"[sdnet] {len(taken)} clean concrete (sampled 1-in-{step} of {len(clean)}) | "
          f"EXCLUDED {len(cracked)} cracked-but-unmasked")


# ------------------------------------------------------------------ Severstal
def adapt_severstal(rows, limit=None):
    limit = limit or 4000
    """Steel, RLE masks for 4 defect classes: scale, patches, inclusions, scratches.
    None are cracks, so defective -> hard negative and clean -> background.
    """
    p = RAW / "kaggle_severstal.zip"
    if not p.exists():
        print("[severstal] SKIP")
        return
    z = zipfile.ZipFile(p)
    defect_ids = set()
    try:
        import csv as _csv
        txt = z.read("train.csv").decode("utf-8", "replace").splitlines()
        for r in _csv.DictReader(txt):
            rle = (r.get("EncodedPixels") or "").strip()
            iid = r.get("ImageId") or r.get("ImageId_ClassId", "").split("_")[0]
            if rle:
                defect_ids.add(iid)
    except Exception as e:  # noqa: BLE001
        print(f"[severstal] could not parse train.csv ({e}); treating all as backgrounds")

    names = [n for n in z.namelist() if "train_images/" in n and n.endswith(".jpg")]
    n_hard = n_bg = 0
    for name in names[:limit]:
        img = imdec(z.read(name))
        if img is None:
            continue
        fn = Path(name).name
        if fn in defect_ids:
            write_pair(rows, "severstal", Path(name).stem, "steel", "B",
                       "hard_negative", img, None)
            n_hard += 1
        else:
            write_pair(rows, "severstal", Path(name).stem, "steel", "C",
                       "negative", img, None)
            n_bg += 1
    print(f"[severstal] {n_hard} steel hard-negatives / {n_bg} clean steel backgrounds")


# ------------------------------------------------------------------ Wood (Kodytek)
WOOD_CRACK_BGR = [(100, 0, 255), (0, 175, 255)]   # #FF0064 Crack, #FFAF00 knot_with_crack
WOOD_OTHER_BGR = [(0, 255, 0), (0, 0, 255), (100, 100, 0), (255, 0, 255),
                  (255, 0, 0), (0, 100, 255), (255, 255, 16), (0, 64, 0)]
WOOD_MAX_SIDE = 1024


def adapt_wood(rows, limit=None):
    """6 image shards paired against the semantic-map archive.

    Only Crack and knot_with_crack become foreground. Knots, resin, marrow, blue
    stain and the rest are real wood defects but not cracks -> hard negatives.
    """
    maps_zip = RAW / "Wood_Semantic_Maps.zip"
    shards = sorted(RAW.glob("Wood_Images*.zip"))
    if not maps_zip.exists() or not shards:
        print("[wood] SKIP (maps or shards missing)")
        return
    print(f"[wood] {len(shards)} shards + semantic maps")

    zmaps = zipfile.ZipFile(maps_zip)
    map_by_id = {Path(n).stem.replace("_segm", ""): n
                 for n in zmaps.namelist() if n.lower().endswith(".bmp")}

    n_pos = n_hard = n_clean = 0
    for sp in shards:
        z = zipfile.ZipFile(sp)
        names = [n for n in z.namelist()
                 if n.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
        print(f"[wood] {sp.name}: {len(names)} images", flush=True)
        for i, n in enumerate(names):
            if limit and (n_pos + n_hard + n_clean) >= limit:
                break
            sid = Path(n).stem
            mn = map_by_id.get(sid)
            if mn is None:
                continue
            img = imdec(z.read(n))
            smap = imdec(zmaps.read(mn))
            if img is None or smap is None:
                continue
            if smap.shape[:2] != img.shape[:2]:
                smap = cv2.resize(smap, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            sm = smap.astype(np.int16)
            crack = np.zeros(sm.shape[:2], bool)
            for c in WOOD_CRACK_BGR:
                crack |= np.abs(sm - c).max(2) <= 30
            other = np.zeros(sm.shape[:2], bool)
            for c in WOOD_OTHER_BGR:
                other |= np.abs(sm - c).max(2) <= 30

            # downscale: originals are ~2800x1024 and we train at 256
            h, w = img.shape[:2]
            if max(h, w) > WOOD_MAX_SIDE:
                s = WOOD_MAX_SIDE / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                crack = cv2.resize(crack.astype(np.uint8), (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                other = cv2.resize(other.astype(np.uint8), (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

            if crack.any():
                write_pair(rows, "wood", sid, "wood", "A", "positive", img,
                           crack.astype(np.uint8) * 255)
                n_pos += 1
            elif other.any():
                write_pair(rows, "wood", sid, "wood", "B", "hard_negative", img, None)
                n_hard += 1
            elif n_clean < 3000:
                write_pair(rows, "wood", sid, "wood", "C", "negative", img, None)
                n_clean += 1
            if i % 500 == 0:
                print(f"    {sp.name} {i}/{len(names)} "
                      f"pos={n_pos} hard={n_hard} clean={n_clean}", flush=True)
        z.close()
    print(f"[wood] {n_pos} crack / {n_hard} non-crack-defect hard-neg / {n_clean} clean")


# ------------------------------------------------------------------ EVAL-ONLY sources
# Small; routed entirely into test splits by dataset/split.py.
EVAL_ONLY_SOURCES = {"kolektor1", "mvtec"}


def adapt_kolektor1(rows, limit=None):
    """KolektorSDD v1: cracks in the epoxy embedding of commutators, with masks.

    Unlike SDD2 these defects are specifically cracks. kosNN/PartM.jpg + _label.bmp.
    """
    p = RAW / "KolektorSDD1.zip"
    if not p.exists():
        print("[kolektor1] SKIP")
        return
    z = zipfile.ZipFile(p)
    imgs = {f"{n.split('/')[0]}_{Path(n).stem}": n
            for n in z.namelist() if n.endswith(".jpg")}
    labs = {f"{n.split('/')[0]}_{Path(n).stem.replace('_label','')}": n
            for n in z.namelist() if n.endswith("_label.bmp")}
    n_pos = n_neg = 0
    for sid, ip in sorted(imgs.items()):
        if limit and n_pos + n_neg >= limit:
            break
        ln = labs.get(sid)
        img = imdec(z.read(ip))
        msk = imdec(z.read(ln), cv2.IMREAD_GRAYSCALE) if ln else None
        if img is None:
            continue
        if msk is not None and (msk > 127).any():
            write_pair(rows, "kolektor1", sid, "plastic", "A", "positive", img, msk)
            n_pos += 1
        else:
            write_pair(rows, "kolektor1", sid, "plastic", "C", "negative", img, None)
            n_neg += 1
    print(f"[kolektor1] {n_pos} REAL plastic cracks / {n_neg} clean  (EVAL-ONLY)")


def adapt_mvtec(rows, limit=None):
    """MVTec-AD crack and scratch subsets (see dataset/fetch_mvtec_cracks.py).

    Scratches get role='scratch', which the binary head reads as background and the
    three-class head reads as a second foreground class.
    """
    base = RAW / "mvtec_cracks"
    idx = base / "index.json"
    if not idx.exists():
        print("[mvtec] SKIP (run dataset/fetch_mvtec_cracks.py first)")
        return
    import json as _json
    n = n_s = 0
    for rec in _json.loads(idx.read_text()):
        if limit and n >= limit:
            break
        img = cv2.imread(str(ROOT / rec["image"]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(ROOT / rec["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        if not (msk > 127).any():
            continue
        defect = rec.get("defect", "crack")
        if defect == "scratch":
            # keep the mask: it is a real annotated scratch, usable as a scratch
            # positive or as a masked hard negative for the crack task
            write_pair(rows, "mvtec", rec["sid"], rec["material"], "B", "scratch",
                       img, msk)
            n_s += 1
        else:
            write_pair(rows, "mvtec", rec["sid"], rec["material"], "A", "positive",
                       img, msk)
            n += 1
    print(f"[mvtec] {n} REAL cracks + {n_s} REAL scratches  (EVAL-ONLY)")


def adapt_steeldefectx(rows, limit=None):
    """SteelDefectX (CC-BY-4.0): real steel cracks plus thin bright/dark scratch
    masks. The scratch masks are usable targets, unlike MVTec's anomaly blobs.
    """
    base = RAW / "steeldefectx"
    idx = base / "index.json"
    if not idx.exists():
        print("[steeldefectx] SKIP (run dataset/fetch_steeldefectx.py first)")
        return
    import json as _json
    n_c = n_s = n_h = 0
    for rec in _json.loads(idx.read_text()):
        if limit and n_c + n_s + n_h >= limit:
            break
        img = cv2.imread(str(ROOT / rec["image"]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(ROOT / rec["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None or not (msk > 127).any():
            continue
        d = rec["defect"]
        if d == "crack":
            write_pair(rows, "sdx", rec["sid"], "steel", "A", "positive", img, msk)
            n_c += 1
        elif d == "scratch":
            write_pair(rows, "sdx", rec["sid"], "steel", "B", "scratch", img, msk)
            n_s += 1
        else:
            # 11 non-crack classes, with masks. Mask dropped for training (these are
            # background) but kept on disk: it shows where a false positive landed.
            write_pair(rows, "sdx", rec["sid"], "steel", "B", "hard_negative", img, None)
            n_h += 1
    print(f"[steeldefectx] {n_c} steel CRACKS / {n_s} steel scratches / "
          f"{n_h} masked steel hard-negatives")


def adapt_casting_impeller(rows, limit=None):
    """Casting impellers from Pilot Technocast, Rajkot (CC-BY-NC-SA-4.0).

    Fixed overhead camera at constant working distance -- the only source matching the
    deployment geometry. Labels are classification only, so ok_front is imported as
    negatives and def_front is staged in data/label_queue/ until it has masks.
    """
    base = RAW / "casting_impeller"
    if not base.exists():
        print("[casting] SKIP (fetch kaggle casting_impeller first)")
        return
    n_ok = 0
    for sub in ("casting_data/casting_data/train/ok_front",
                "casting_data/casting_data/test/ok_front"):
        d = base / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpeg")) + sorted(d.glob("*.jpg")):
            if limit and n_ok >= limit:
                break
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            write_pair(rows, "casting", p.stem, "steel", "A", "negative", img, None)
            n_ok += 1

    # Copied, not referenced, so the queue can be handed to someone as-is.
    queue = ROOT / "data/label_queue/casting"
    queue.mkdir(parents=True, exist_ok=True)
    n_q = 0
    for sub in ("casting_512x512/casting_512x512/def_front",
                "casting_data/casting_data/train/def_front",
                "casting_data/casting_data/test/def_front"):
        d = base / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpeg")) + sorted(d.glob("*.jpg")):
            dest = queue / f"{d.parent.name}_{p.stem}.png"
            if dest.exists():
                n_q += 1
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if not cv2.imwrite(str(dest), img):
                raise IOError(f"failed to write {dest}")
            n_q += 1
    print(f"[casting] {n_ok} clean fixed-camera negatives imported / "
          f"{n_q} defective images staged in data/label_queue/casting (UNLABELLED, "
          f"not in the corpus)")


ADAPTERS = {
    "steeldefectx": adapt_steeldefectx,
    "casting_impeller": adapt_casting_impeller,
    "crackseg9k": adapt_crackseg9k,
    "magnetic_tile": adapt_magnetic_tile,
    "neu": adapt_neu,
    "wood": adapt_wood,
    "kolektor1": adapt_kolektor1,
    "mvtec": adapt_mvtec,
    "kolektor": adapt_kolektor,
    "gc10": adapt_gc10,
    "sdnet": adapt_sdnet,
    "severstal": adapt_severstal,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, help="cap images per source (smoke test)")
    args = ap.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MSK_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, fn in ADAPTERS.items():
        if args.only and name not in args.only:
            continue
        fn(rows, args.limit)

    # background-only sources (no masks, never positives)
    if not args.only or "dtd" in (args.only or []):
        _flat_images(RAW / "dtd-r1.0.1.tar.gz", "dtd", "mixed", "negative", "C",
                     rows, args.limit, exclude=("cracked",))   # real cracks, no masks
    if not args.only or "ozgenel" in (args.only or []):
        _flat_images(RAW / "kaggle_concrete_ozgenel.zip", "ozgenel", "concrete",
                     "negative", "C", rows, args.limit or 6000, exclude=("positive",))

    # Merge with the existing index so adapters can be run incrementally.
    out = CLEAN / "index_raw.csv"
    if rows:
        fields = list(rows[0])
        existing = []
        if out.exists():
            with out.open(newline="", encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh)]
            new_names = {r["name"] for r in rows}
            existing = [r for r in existing if r["name"] not in new_names]
        with out.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(existing + rows)
        print(f"\nwrote {len(rows)} new + {len(existing)} existing = "
              f"{len(rows)+len(existing)} rows -> {out}")

if __name__ == "__main__":
    main()
