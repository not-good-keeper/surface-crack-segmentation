"""Fetch only MVTec-AD's crack-labelled images and masks (~50 MB, not the 4.9 GB
archive -- the official link is dead, so this uses the Voxel51 HF mirror).

Keeps `tile` and `capsule`; drops `pill` and `hazelnut`, which are real cracks on
things that are not industrial surfaces. Eval-only, CC BY-NC-SA 4.0.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/mvtec_cracks"
REPO = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main"
# crack classes -> materials
KEEP = {"tile": "ceramic", "capsule": "plastic"}
# scratch classes -> materials. screw/metal_nut are steel-family surfaces, which
# is the material with no real crack data at all, so real scratches there are
# doubly useful: a defect class we promised AND a surface we cannot otherwise test.
SCRATCH_KEEP = {"metal_nut": "metal", "capsule": "plastic",
                "wood": "wood", "screw": "metal"}
SKIP_NOTE = {"pill", "hazelnut"}


def get(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=90, allow_redirects=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {url}: {e}")
        return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sj = OUT / "samples.json"
    if not sj.exists():
        print("[mvtec] fetching samples.json ...")
        if not get(f"{REPO}/samples.json", sj):
            return 1

    data = json.loads(sj.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    def pick(defect_pred, keep):
        return [s for s in samples
                if defect_pred(str((s.get("defect") or {}).get("label", "")))
                and (s.get("category") or {}).get("label") in keep]

    cracks = pick(lambda d: d == "crack", KEEP)
    scratches = pick(lambda d: d.startswith("scratch"), SCRATCH_KEEP)
    skipped = [s for s in samples
               if (s.get("defect") or {}).get("label") == "crack"
               and (s.get("category") or {}).get("label") in SKIP_NOTE]

    print(f"[mvtec] {len(scratches)} in-domain SCRATCH samples "
          f"({', '.join(sorted(SCRATCH_KEEP))})")
    print(f"[mvtec] {len(cracks)} in-domain crack samples "
          f"({', '.join(sorted(KEEP))}); skipping {len(skipped)} out-of-domain "
          f"({', '.join(sorted(SKIP_NOTE))})")

    jobs, index = [], []
    for s, defect_class, keep in ([(x, 'crack', KEEP) for x in cracks] +
                                  [(x, 'scratch', SCRATCH_KEEP) for x in scratches]):
        cat = s["category"]["label"]
        img_rel = s["filepath"]
        msk_rel = s["defect_mask"]["mask_path"]
        sid = f"{defect_class}_{cat}_{Path(img_rel).stem}"
        ip = OUT / "images" / f"{sid}.png"
        mp = OUT / "masks" / f"{sid}.png"
        jobs.append((f"{REPO}/{img_rel}", ip))
        jobs.append((f"{REPO}/{msk_rel}", mp))
        index.append(dict(sid=sid, category=cat, defect=defect_class,
                          material=keep[cat],
                          image=str(ip.relative_to(ROOT)),
                          mask=str(mp.relative_to(ROOT))))

    print(f"[mvtec] downloading {len(jobs)} files ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(lambda j: get(*j), jobs))

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    got = sum(1 for i in index
              if (ROOT / i["image"]).exists() and (ROOT / i["mask"]).exists())
    print(f"[mvtec] complete pairs: {got}/{len(index)}  ({sum(ok)}/{len(jobs)} files)")
    by = {}
    for i in index:
        by[i["material"]] = by.get(i["material"], 0) + 1
    print(f"[mvtec] by material: {by}")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
