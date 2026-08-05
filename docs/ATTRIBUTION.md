# Attribution and licences

Every source in the corpus, its licence, and what was taken from it. Downloads are
pinned by sha256 in `data/raw/.lock.json`; Roboflow fetches record project, version and
export format in `data/raw/roboflow/manifest.json`.

**Licence compatibility.** The corpus mixes CC0, CC-BY, CC-BY-NC and research-only
terms. The most restrictive terms present are **non-commercial** (MVTec, KolektorSDD2,
the casting set, the sanitary-ceramics sets) and **GPL-3.0** (MSD, if pulled). Any
release beyond the hackathon must re-check these per source; a trained model derived
from NC data is not automatically free of the NC condition.

---

## 1. Pixel-mask sources used as segmentation targets

| Source | Licence | Credit | Taken |
|---|---|---|---|
| **CrackSeg9k v4** | CC0-1.0 | Kulkarni et al. — doi:10.7910/DVN/EGIEBY | 7,570 crack masks; asphalt, plaster, concrete, masonry, ceramic |
| **Wood semantic maps** | CC-BY-4.0 | Kodytek, Bodzas, Bilik — doi:10.5281/zenodo.4694695 | 1,156 wood crack masks; `knot_with_crack` excluded |
| **SteelDefectX** | CC-BY-4.0 | Zhaosxian — HuggingFace `Zhaosxian/SteelDefectX` | 300 steel cracks, 763 steel scratches, 4,499 masked hard negatives |
| **Magnetic Tile** | see repo | Huang et al. — github.com/abin24/Magnetic-tile-defect-datasets | ceramic tile cracks + clean tiles |
| **KolektorSDD1** | research | Tabernik et al., ViCoS — vicos.si/resources/kolektorsdd | 49 epoxy crack masks |
| **MVTec AD** | CC-BY-NC-SA-4.0 | Bergmann et al., MVTec Software GmbH | 17 tile cracks; scratches quarantined; capsule reclassified |

### Roboflow Universe (rasterised from COCO polygons)

Each is CC-BY-4.0 as published on Universe. Fetched via `dataset/fetch_roboflow.py`,
rasterised by `dataset/rf_adapt.py`.

| Project | Credit | Kept | Notes |
|---|---|---|---|
| `gazxard/pipe-crack-detection` v1 | gazxard | **937 plastic** | `PVC pipe crack` only — `Paper crack` (957) and `Dummy crack` (954) rejected as out of domain |
| `steel-reuse/steel-reuse-cracks-ysu8b` v1 | steel-reuse | **293 steel** | passed the width audit at 8.75 px median |
| `aleksandr-endoscope/endoscope-cracks-segmentation` v3 | aleksandr-endoscope | **131 steel** | borescope optics, not conveyor capture — flagged |
| `heechan/cracked_phone` v4 | heechan | **12 glass** | 19 of 31 rejected as whole-object or blob masks |

---

## 2. Negative and hard-negative sources

| Source | Licence | Credit | Taken |
|---|---|---|---|
| **Casting product images** | CC-BY-NC-SA-4.0 | Ravirajsinh Dabhi; imagery from Pilot Technocast, Rajkot, Gujarat | 3,137 clean fixed-camera negatives; 4,992 defective staged unlabelled |
| Severstal Steel | competition terms | Severstal / Kaggle | 2,078 hard negatives, 1,920 clean steel |
| NEU-DET | research | Song & Yan, Northeastern University | 1,499 steel hard negatives; `crazing` excluded (boxes only) |
| GC10-DET | research | Lv et al. | 2,300 metal hard negatives |
| KolektorSDD2 | CC-BY-NC-SA-4.0 | Božič, Tabernik, Skočaj — ViCoS | 2,979 clean industrial surfaces |
| SDNET2018 | open | Dorafshan, Thomas, Maguire — Utah State | 6,000 clean concrete |
| Özgenel surface crack | open | Özgenel & Sorguç | 5,830 clean concrete |
| Describable Textures (DTD) | research | Cimpoi et al., VGG Oxford | 5,464 textures; `cracked` category excluded |
| `demoworkspace-0gyhr/damage-detection-on-display` | CC-BY-4.0 | demoworkspace | quarantined: coarse blob masks, imported detection-only |

---

## 3. Identified, reviewed, not used

| Source | Why not |
|---|---|
| `dataset-rmy16/glass-dataset-scratch` | its 1,443 polygons annotate the **glass panel region**, not the defect; the actual scratch annotations carry no polygons |
| `instacash/mobile-minute-cracks` | images are **line-art schematics of phones**, not photographs of a surface |
| `yolo-r8fla/glass-detection-uymwf` | only 3 rows survive the polygon filter, all near-black and unverifiable |
| `molding-2avtd/molding-kabyb` | 9,956 images but **bounding boxes only**, no polygons |
| `spencerworkspace/ceramic-tile-surface-defects` | project type is **classification**, no masks |
| `h-rnzqa/cracks-n5ahr`, `ceramic/ceramic-defect-detection`, `shahin-workspace/metal-defects-jn5zf`, `mobile-scratch/mobile-cracks-segmentation` | browsable on Universe but have **no generated version**, so they cannot be downloaded through the API. `metal-defects` is the painful one: 1,098 instance-segmentation images with 781 crack annotations, and non-steel metal remains our weakest material |
| Mendeley `47x6jdbr5j` (Nascimento et al., CC-BY-4.0) | 1,600 ceramic tile images but **classification labels only** |
| Tianchi aluminium profile | registration gated on a Chinese mobile number |
| Injection-moulded polypropylene | embargoed until end-2027 |
| OmniCrack30k | licence-gated, request pending |
| Culvert-Sewer Defects | author-gated |

### Available and worth pulling next

**Surface defect dataset for sanitary ceramics** (Mendeley `wmptdnw356` v1 and
`2bkhytgwm8` V2), CC-BY-NC-3.0 — Murat Çöpoğlu, Gürkan Öztürk, Emre Çimen, Eskişehir
Teknik Üniversitesi. Ships genuine pixel masks on ceramic sanitary ware, which is
directly on-target for the Morbi cluster.

One caution that governs how it must be ingested: its 18,560 patches derive from only
**106 original photographs**. Grouping by parent photograph is mandatory — a naive split
would put augmentations of one photo on both sides and inflate every number reported.
The two versions also appear to overlap and must be de-duplicated rather than stacked.

---

## 4. How to cite this corpus

The corpus is an assembly, not a new dataset. Cite the sources above directly. If the
assembly itself needs a reference, state the split version and hash so the exact
partition can be reproduced — `dataset/split.py` regenerates it from the image files.
