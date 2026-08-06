"""Build the release archive for Hugging Face / Kaggle.

    python dataset/package.py                 # metadata + masks + pipeline (small)
    python dataset/package.py --samples 40    # ...plus sample images per open source

Ships what we are entitled to ship and can prove: the annotations, the manifest, the
split freeze, the rebuild pipeline and the documentation. Third-party photographs are
NOT redistributed; the archive rebuilds them from their original hosts instead.

Two reasons, and the first is not negotiable:

  Licensing.  Only 7 of 18 sources may be redistributed -- 18,843 of 51,504 rows
  (36.6 %). DTD is research-only; KolektorSDD2, casting and MVTec are CC-BY-NC-SA, and
  share-alike means bundling any of them drags the whole derived corpus to NC-SA; six
  more had no licence recorded at fetch and an unverified licence is a licence we do not
  have. A single archive containing them would be undistributable as a whole, and
  splitting the difference silently is how a dataset ends up quietly unusable by anyone
  who checks.

  Size.  The image corpus is 18.5 GB. The annotations are 60 MB.

Masks are included only for sources whose licence permits it. A mask over a photograph
is a derivative work of that photograph, so an NC source's mask inherits NC -- shipping
it would re-import the problem this script exists to avoid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
OUT = ROOT / "data/release"

def licence_of(source: str, reg: dict) -> tuple[str, bool]:
    """-> (licence string, may we redistribute), from dataset/licences.yaml.

    Keyed by the manifest's own `source` values rather than inferred from
    sources.yaml, which registers only the seven sources fetched by fetch.py. Reading
    the incomplete file resolved fourteen of eighteen sources to "unknown" -- safe in
    direction, but it would have withheld 4,755 rows of CC-BY-4.0 SteelDefectX and
    described CC-BY-NC-SA casting imagery as merely unverified.

    A source absent from the registry is restricted, not assumed open.
    """
    e = reg.get(source)
    if not isinstance(e, dict):
        return "unregistered", False
    return str(e.get("licence", "unknown")), bool(e.get("redistribute", False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0,
                    help="sample images per openly-licensed source (0 = none). "
                         "Images are the 18.5 GB; keep this small")
    ap.add_argument("--name", default="conveyor-surface-defects-v7")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data/manifest_split.csv")
    reg = yaml.safe_load((ROOT / "dataset/licences.yaml").read_text()) or {}

    verdict = {s: licence_of(s, reg) for s in sorted(df.source.astype(str).unique())}
    open_sources = {s for s, (_, ok) in verdict.items() if ok}

    print(f"{'source':<24}{'licence':<22}{'rows':>7}  redistribute")
    for s, (lic, ok) in sorted(verdict.items(), key=lambda kv: (not kv[1][1], kv[0])):
        print(f"{s:<24}{lic:<22}{int((df.source == s).sum()):>7}  "
              f"{'yes' if ok else 'NO -- rebuild from source'}")

    n_open = int(df.source.isin(open_sources).sum())
    print(f"\n{n_open} / {len(df)} rows are openly licensed "
          f"({100 * n_open / max(len(df), 1):.1f} %)")

    OUT.mkdir(parents=True, exist_ok=True)
    staging = OUT / args.name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # ---- metadata: every row, including restricted ones ---------------------
    # The manifest describes images; it does not contain them. Measurements, split
    # assignment and provenance are facts about the corpus and are what make the
    # rebuild reproducible, so restricted sources stay listed -- with a column saying
    # plainly that their pixels are not in this archive.
    meta = df.copy()
    meta["licence"] = meta.source.map(lambda s: verdict.get(str(s), ("unknown", 0))[0])
    meta["images_included"] = meta.source.isin(open_sources)
    meta.to_csv(staging / "manifest_split.csv", index=False)

    # ---- annotations --------------------------------------------------------
    masks_dir = staging / "masks"
    masks_dir.mkdir()
    n_masks = 0
    for name in meta[meta.images_included].name:
        src = CLEAN / "masks" / f"{name}.png"
        if src.exists():
            shutil.copy2(src, masks_dir / f"{name}.png")
            n_masks += 1

    # ---- optional sample images --------------------------------------------
    n_imgs = 0
    if args.samples:
        img_dir = staging / "images_sample"
        img_dir.mkdir()
        for s in sorted(open_sources):
            sub = meta[(meta.source == s)].head(args.samples)
            for name in sub.name:
                src = CLEAN / "images" / f"{name}.png"
                if src.exists():
                    shutil.copy2(src, img_dir / f"{name}.png")
                    n_imgs += 1

    # ---- the rebuild pipeline ----------------------------------------------
    pipe = staging / "pipeline"
    pipe.mkdir()
    for f in ("sources.yaml", "licences.yaml", "fetch.py", "adapters.py",
              "normalize.py", "split.py", "qa.py", "index.py"):
        p = ROOT / "dataset" / f
        if p.exists():
            shutil.copy2(p, pipe / f)

    docs = staging / "docs"
    docs.mkdir()
    for f in ("DATASET.md", "DATASET_SCORECARD.md", "ATTRIBUTION.md"):
        p = ROOT / "docs" / f
        if p.exists():
            shutil.copy2(p, docs / f)
    rep = ROOT / "data/report/normalize.txt"
    if rep.exists():
        shutil.copy2(rep, docs / "normalize_report.txt")

    (staging / "LICENCES.json").write_text(json.dumps(
        {s: {"licence": lic, "images_redistributed": ok,
             "rows": int((df.source == s).sum()),
             "credit": (reg.get(s) or {}).get("credit"),
             "reason": (reg.get(s) or {}).get("reason")}
         for s, (lic, ok) in verdict.items()}, indent=1))

    write_card(staging, meta, verdict, open_sources, n_masks, n_imgs, args.name)

    # ---- zip ----------------------------------------------------------------
    archive = OUT / f"{args.name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(staging.parent))

    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    mb = archive.stat().st_size / 1e6
    print(f"\nmasks {n_masks} | sample images {n_imgs}")
    print(f"-> {archive}  ({mb:.1f} MB)")
    print(f"   sha256 {sha}")
    (OUT / f"{args.name}.sha256").write_text(f"{sha}  {archive.name}\n")
    return 0


def write_card(staging, meta, verdict, open_sources, n_masks, n_imgs, name):
    """The Hugging Face dataset card. Front-matter first, then the human part."""
    splits = meta.groupby("split").size().sort_values(ascending=False)
    restricted = [(s, lic) for s, (lic, ok) in sorted(verdict.items()) if not ok]
    rows = "\n".join(f"| `{s}` | {int(n)} |" for s, n in splits.items())
    restr = "\n".join(f"| `{s}` | {lic} |" for s, lic in restricted)
    n_open = int(meta.images_included.sum())

    (staging / "README.md").write_text(f"""---
license: cc-by-4.0
task_categories:
- image-segmentation
tags:
- industrial
- defect-detection
- crack-segmentation
- manufacturing
- quality-inspection
size_categories:
- 10K<n<100K
---

# Conveyor-Line Surface Defect Segmentation (v7)

Three-class surface-defect segmentation for end-of-line manufacturing inspection:
**background / crack / scratch**, mutually exclusive through a softmax.

Built for a fixed overhead camera on a conveyor belt inspecting finished products. The
system reports *where* a defect is and *what type* it is, with per-region geometry. It
deliberately emits **no pass/fail verdict** — the machine searches, a human decides.

## What is in this archive

| contents | included |
|---|---|
| `manifest_split.csv` | **all {len(meta)} rows** — measurements, provenance, frozen split |
| `masks/` | {n_masks} annotation masks (openly-licensed sources) |
| `images_sample/` | {n_imgs} sample images |
| `pipeline/` | the scripts that rebuild the corpus from source |
| `docs/` | strategy, scorecard, attribution, normalisation audit |

## What is NOT in this archive, and why

**Photographs are not redistributed.** The corpus is 18.5 GB and, more importantly,
three of its eighteen sources restrict redistribution:

| source | licence |
|---|---|
{restr}

DTD is research-only; KolektorSDD2 is non-commercial **and** share-alike, so bundling it
would force this entire derived dataset to CC-BY-NC-SA; `magnetic_tile`'s licence was
never verified, and an unverified licence is a licence we do not have.

Rather than ship a corpus nobody can legally reuse, the archive ships the **pipeline**.
`{n_open}` of {len(meta)} rows come from openly-licensed sources and their masks are
included; every row is described in the manifest whether or not its pixels are here.

```bash
python pipeline/fetch.py          # download originals from their own hosts
python pipeline/adapters.py       # normalise to image/mask pairs
python pipeline/normalize.py      # width audit, de-duplication, material refinement
python pipeline/split.py          # regenerate the frozen split
python pipeline/qa.py --strict    # leakage and integrity gates -- must pass
```

## Splits

Grouped by **parent image**, so crops of one photograph cannot straddle a split
boundary, and frozen so a result always names the freeze it was measured under.

| split | n |
|---|---:|
{rows}

`test_scratch_blob` reports **detection and class only**. Its masks have a 33.6 px
median width against 4–15 px for every accepted thin-structure source, so scoring
geometry against it would measure agreement with a mask convention we rejected.

`val_unseen_material` is separate from `test_unseen_material` on purpose: selecting an
epoch on the same material you report transfer on is not transfer measurement.

## Known limitations

Stated because a dataset card that omits them is worth less than no card.

- **Scratches are one material.** 724 real scratch images, all steel. Any scratch claim
  on ceramic, plastic or glass is transfer, not measurement.
- **The headline test split is 156 images.** Differences under ~0.03 clDice are noise.
- **No primary capture.** Nothing here was photographed on a conveyor under fixed
  industrial lighting; camera realism is simulated, not sampled.
- **Thin factory coverage**: ceramic 168, plastic 935, epoxy 49, glass 12.

See `docs/DATASET_SCORECARD.md` for the full accounting, including every rejected source
and the measurement that justified rejecting it.

## Citation

Please cite the **original sources** — listed with licences in `LICENCES.json` and
`docs/ATTRIBUTION.md`. This release contributes the annotations, the normalisation and
de-duplication pipeline, the frozen split design and the QA gates; it does not claim
authorship of the underlying photographs.
""", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
