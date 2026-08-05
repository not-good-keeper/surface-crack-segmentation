"""Fetch the crack and scratch classes of SteelDefectX (CC-BY-4.0).

Three prefixes only: `cracking`, and `bs`/`ds` (bright/dark scratches). The other 22
classes are steel defects that are neither, and steel hard negatives are not scarce.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/steeldefectx"
API = "https://huggingface.co/api/datasets/Zhaosxian/SteelDefectX"
RAW = "https://huggingface.co/datasets/Zhaosxian/SteelDefectX/resolve/main"

# filename prefix -> (our defect class, material)
WANT = {
    "cracking": ("crack", "steel"),
    "bs": ("scratch", "steel"),
    "ds": ("scratch", "steel"),
}

# The remaining classes are real steel defects that are NOT cracks. They make far better
# hard negatives than the mask-less ones we had (NEU-DET, Severstal, GC10), because a
# masked negative lets us measure exactly where the model wrongly fires. Enabled with
# --hard-negatives.
HARD_NEG = {
    "in": "inclusion", "pa": "patches", "ps": "pitted_surface", "rs": "rolled_in_scale",
    "cg": "crescent_gap", "wl": "welding_line", "ws": "water_spot", "os": "oil_spot",
    "ss": "silk_spot", "pu": "punching", "ris": "red_iron_sheet",
}


def prefix_of(stem: str) -> str:
    # 'cracking_12' -> 'cracking' ; 'bs_003' -> 'bs'
    return stem.split("_")[0].lower()


def get(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=90, allow_redirects=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hard-negatives", action="store_true",
                    help="also fetch the 11 non-crack steel classes")
    ap.add_argument("--cap", type=int, default=90, help="per hard-neg class")
    args = ap.parse_args()

    files = [f["rfilename"] for f in requests.get(API, timeout=60).json()["siblings"]]
    pairs = []
    for split in ("train", "val"):
        imgs = [f for f in files if f.startswith(f"{split}/")]
        masks = {Path(f).stem: f for f in files if f.startswith(f"{split}_mask/")}
        for f in imgs:
            stem = Path(f).stem
            pre = prefix_of(stem)
            if pre not in WANT or stem not in masks:
                continue
            cls, mat = WANT[pre]
            pairs.append((f"{split}_{stem}", cls, mat, f, masks[stem]))

    if args.hard_negatives:
        from collections import Counter
        seen = Counter()
        for split in ("train", "val"):
            masks = {Path(f).stem: f for f in files if f.startswith(f"{split}_mask/")}
            for f in [x for x in files if x.startswith(f"{split}/")]:
                stem = Path(f).stem
                pre = prefix_of(stem)
                if pre not in HARD_NEG or stem not in masks:
                    continue
                if seen[pre] >= args.cap:
                    continue
                seen[pre] += 1
                pairs.append((f"{split}_{stem}", "hard_negative", "steel",
                              f, masks[stem]))
        print(f"[sdx] + {sum(seen.values())} masked steel hard negatives "
              f"across {len(seen)} classes")

    from collections import Counter as _C
    by_cls = _C(p[1] for p in pairs)
    print(f"[sdx] {len(pairs)} pairs: " + " / ".join(
        f"{n} {c}" for c, n in sorted(by_cls.items())))

    jobs, index = [], []
    for sid, cls, mat, irel, mrel in pairs:
        ip, mp = OUT / "images" / f"{sid}.png", OUT / "masks" / f"{sid}.png"
        jobs += [(f"{RAW}/{irel}", ip), (f"{RAW}/{mrel}", mp)]
        index.append(dict(sid=sid, defect=cls, material=mat,
                          image=str(ip.relative_to(ROOT)),
                          mask=str(mp.relative_to(ROOT))))

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[sdx] downloading {len(jobs)} files ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda j: get(*j), jobs))

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    got = sum(1 for i in index if (ROOT / i["image"]).exists()
              and (ROOT / i["mask"]).exists())
    print(f"[sdx] complete pairs: {got}/{len(index)}   licence CC-BY-4.0")
    return 0 if got else 1


if __name__ == "__main__":
    main()
