# Dataset Strategy, Provenance and Expansion Plan

## 1. Why this dataset needs a deliberate strategy

The problem is factory end-of-line surface inspection, not generic civil crack detection. The initial corpus was useful for learning thin crack morphology, but most of its labelled cracks come from asphalt, plaster, concrete, masonry and wood. Those images are valuable auxiliary training and transfer evidence; they are not proof that a model works on factory products.

The dataset is therefore organised around one rule: **factory steel, metal, ceramic, plastic and epoxy are the target domain; civil materials are auxiliary evidence.** We retain civil data because its varied crack geometry improves representation learning, but report factory and cross-material results separately and state the material gaps openly.

The final task is three-class pixel segmentation—background, crack and scratch—with clean/hard-negative images used to control false positives. A detector can be explored later, but it is not a substitute for a mask and cannot provide region geometry.

## 1a. Target products, and where they actually are

The deployment scenario is a fixed overhead camera on a conveyor inspecting finished products before packaging. That narrows the product list considerably: it must move on a line, present a roughly flat inspectable face to a camera directly above it, and fail on crack or scratch rather than on assembly or dimensional error. The list below is drawn from India's real MSME manufacturing clusters under those three constraints.

| Product | Cluster | Line-relevant defects | Data status |
|---|---|---|---|
| Stainless-steel castings, pump impellers | Rajkot, Coimbatore — roughly 5,000 foundries nationally, overwhelmingly MSME | shrinkage cracks, machining scratches, blowholes | **real fixed-camera images held**, unlabelled (§4a) |
| Rolled and fabricated steel sheet, strip, stamped parts | Ludhiana, Rajkot, Faridabad, Jamshedpur | edge cracks, bright and dark scratches, inclusions | **real masks held** (SteelDefectX) |
| Ceramic floor and wall tiles, sanitary ware | **Morbi, Gujarat — 459 operating units, roughly 70 % of national tile output, ~68,000 direct employees**, including 43 sanitary-ware units | glaze crack, hairline crack, chip, scratch | partial masks (168) |
| Auto components, painted panels | Pune, Chennai, Gurgaon–Manesar | paint scratch, clear-coat crack | no public masks |
| Plastic mouldings, electrical housings, switch plates | Daman, Vapi, Noida | stress cracks, scratches, sink marks | **no real crack masks** |
| Glassware, mirrors, tempered and screen glass | Firozabad; electronics assembly at Noida and Sriperumbudur | edge chip, crack, scratch | one GPL-3.0 source identified, not pulled |
| Hand tools, bicycle and cycle parts | Jalandhar, Ludhiana | forging cracks, polishing scratches | no public masks |

**Steel is the first deployment target** despite being the most heavily researched material in the literature. The reason is specific to this project rather than to novelty: it is the only material for which we hold real pixel masks of *both* defect classes, and it is the largest of the clusters above. Ceramic is the strongest second candidate — high Indian relevance, far less saturated research — but it needs production-line captures before any claim is made for it, and Morbi is where those captures would come from.

Two products deliberately **excluded** despite appearing in surface-defect literature: solar panels, because microcrack inspection needs electroluminescence rather than RGB, and pipe or bore interiors, because they need a borescope rather than an overhead fixed camera. Both are different sensing problems, not variations of this one.

## 2. Dataset principles and provenance

Every canonical real-image row includes an image, a binary PNG mask, source ID, source sample ID, material, role, parent group and perceptual hash. Downloaded source files are locked in `data/raw/.lock.json` with SHA-256. The real evaluation suite is frozen in `data/splits.json` v7, SHA-256:

`c0fde17c96749567…` (full value in `data/splits.json`)

**v5 onward is regenerated entirely from source.** Under v4 it was not: the `test_scratch` and `test_steel` splits, and the plaster/epoxy material corrections, existed only as edits applied to the split manifest by no committed script, so `manifest_clean.csv` and `manifest_split.csv` disagreed about what a row was made of and neither could be rebuilt. Those rules now live in `dataset/normalize.py` and `dataset/split.py`, and the four-command sequence in §10 reproduces the frozen suite from the image files.

The role of an image is more important than its filename:

| Role | Meaning | Foreground in segmentation? | Allowed as synthetic substrate? |
|---|---|---:|---:|
| `positive` | Verified crack with a usable pixel mask | Yes | Only from crack-free crops, with mask/margin checks |
| `scratch` | Verified scratch with a usable thin mask | Yes, in the three-class model | No |
| `hard_negative` | Real line-like distractor or non-crack defect: scratch, seam, pit, scale, knot, etc. | Never | No |
| `negative` | Defect-free surface by source construction | No | Yes, after split and quality filtering |

Hard negatives are intentionally retained. An inspection model that labels every seam, wood grain, scratch or pit as a crack will create costly over-rejection—the exact failure this project is meant to reduce.

## 3. Current real-image corpus and coverage

The live corpus contains **51,504 real images**: **10,496 crack masks**, **724 scratch masks**, **11,531 hard negatives**, and **28,753 clean negatives**. Synthetic data is stored separately and is never part of validation or final testing.

| Material | Real crack masks | Coverage interpretation |
|---|---:|---|
| Asphalt | 3,627 | Auxiliary civil source |
| Plaster/render | 3,161 | Auxiliary civil source |
| Wood | 1,156 | Final unseen-material transfer test |
| **Plastic** | **936** | Roboflow PVC pipe; **gap closed**, but single-product |
| Steel | 685 | Primary factory material; 300 SteelDefectX + 293 Roboflow + 92 other |
| Concrete | 442 | Thin auxiliary source |
| Masonry | 229 | Unseen-validation material |
| Ceramic | 168 | Primary factory material, thin coverage |
| Epoxy | 49 | Primary factory-adjacent material, very thin coverage |
| Pharma capsule | 20 | Out of domain; excluded from factory metrics (see below) |
| Glass | 12 | Barely present; any glass claim is transfer, not evidence |
| Non-steel metal | 0 | **Critical data gap** |

Scratch masks are still concentrated on one material: steel 624, non-steel metal 67, wood 21, plastic 12. Every scratch claim on ceramic, plastic or glass rests on transfer plus synthetic data, and must be reported as such.

The two changes since v5 are worth stating explicitly, because they move a headline number. **Plastic went from 0 to 936 real crack masks** through the Roboflow PVC-pipe import, which is the single largest coverage gain in the project — but all 936 are one product photographed one way, so it closes the *zero* problem, not the *diversity* problem. **Glass went from 0 to 12**, which closes nothing; 12 masks cannot support a per-material metric and are reported with the count attached or not at all.

**Correction carried into v5: plastic has zero real crack masks, not 20.** The 20 previously counted as plastic are MVTec `capsule` — photographs of pharmaceutical pills, which are neither a plastic moulding nor a surface. §6 already listed the capsule class as out of domain, but its material label still read `plastic`, so the count contradicted the decision and the images sat in the factory test split. They are now labelled `pharma_capsule` in `dataset/normalize.py`, which excludes them from `FACTORY_MATERIALS` automatically.

### Split v7 — 80:15:5 stratified (current)

Ratios are applied **at group level, stratified by material × foreground quartile**.
Stratifying matters more than the ratio: a pooled shuffle would let one material land
mostly in test and another mostly in train by chance, and with materials as thin as
glass (12 masks) or epoxy (49) that decides whether a per-material number exists at all.

| Split | Images | Defect masks | Materials |
|---|---:|---:|---|
| `train` | 37,438 | 7,187 | all trainable |
| `val` | 7,144 | 1,389 | all trainable |
| `val_unseen_material` | 229 | 229 | masonry |
| **`test_factory`** | **156** | **156** | ceramic 25, epoxy 49, glass 2, plastic 46, steel 34 |
| **`test_factory_scratch`** | **31** | **31** | steel |
| `test_scratch_blob` | 100 | 100 | quarantined MVTec |
| `test_seen` | 379 | 379 | asphalt, concrete, plaster |
| `test_unseen_material` | 1,156 | 1,156 | wood |
| `test_negatives` | 4,626 | 0 | 8 materials |
| `wood_bg` | 245 | 0 | wood |

Plastic and glass appear in a test split here for the first time. The trade-off of a
5 % test share is stated plainly: `test_factory_scratch` falls to **31 images**, so its
clDice carries a wide interval and should be quoted with the count attached.

### Split v5 (superseded, retained for comparison)

| Frozen split (v5) | Images | Crack | Scratch | Purpose |
|---|---:|---:|---:|---|
| `train` | 31,736 | 5,562 | 374 | Model fitting; real samples only, augmented at runtime |
| `val` | 3,591 | 965 | 94 | In-domain training sanity check |
| `val_unseen_material` | 229 | 229 | 0 | Masonry; model selection without contaminating final tests |
| **`test_factory`** | **164** | **164** | 0 | **Headline**: steel 75, epoxy 49, ceramic 40 |
| **`test_factory_scratch`** | **156** | 0 | **156** | **Headline**: held-out real steel scratches |
| `test_scratch_blob` | 100 | 0 | 100 | MVTec scratches; detection and class only, never IoU/clDice |
| `test_seen` | 1,087 | 1,087 | 0 | Civil materials present in training; auxiliary evidence |
| `test_unseen_material` | 1,156 | 1,156 | 0 | Wood transfer test, never trained on as positives |
| `test_negatives` | 5,917 | 0 | 0 | False-positive-area test on clean surfaces |
| `wood_bg` | 245 | 0 | 0 | Synthesis-only wood backgrounds, isolated at board level |

Two defects in v4 that v5 fixes, both of which would have silently broken the three-class model:

1. **Every one of the 879 real scratch images was in a test split.** The scratch class had no real training data at all, so a three-class model would have learned it from synthetic patches and been evaluated on real ones. `sdx` defect rows are now split 60/15/25 across train, val and test.
2. **`parent_id()` collapsed all 763 real scratches onto two group keys** (`sdx:bs`, `sdx:ds`). With two groups the class could only ever land wholly in train or wholly in test — which is how defect 1 arose. `sdx` rows are now grouped per image.

Neither was caught, because `test_scratch` and `test_steel` were absent from `EVAL_SPLITS` in `dataset/qa.py`, and that list is what the leakage check intersects against — an omitted split is not merely unreported, it is unchecked. QA now fails if the manifest contains any split it does not know about, so the general form of this mistake cannot recur.

## 4. Source catalogue: accepted crack and scratch masks

| Source | Current contribution | Why it is included | Important limitations and handling | Reference |
|---|---|---|---|---|
| CrackSeg9k v4 | 8,903 images: 7,570 cracks and 1,333 clean; asphalt, plaster, concrete, masonry and ceramic | Curated paired masks, CC0 licence and broad thin-crack morphology make it the auxiliary backbone | It is mainly civil, so it cannot substantiate a factory claim. `cracktree` is removed; Rissbilder and Volker are correctly relabelled as plaster/render, not concrete. | [Harvard Dataverse, DOI 10.7910/DVN/EGIEBY](https://doi.org/10.7910/DVN/EGIEBY) |
| Kodytek wood semantic maps | 3,852 images: 1,156 cracks, 1,500 hard negatives, 1,196 clean | Pixel semantic maps let us distinguish cracks from knots, stains and grain; CC-BY-4.0 | Auxiliary material and final unseen transfer test. `knot_with_crack` masks outline knots rather than cracks, so they are not foreground. | [Zenodo, DOI 10.5281/zenodo.4694695](https://doi.org/10.5281/zenodo.4694695) |
| SteelDefectX | 5,562 images: 300 steel cracks, 763 thin scratches, 4,499 masked hard negatives | Best current factory source: real steel cracks and thin masks under CC-BY-4.0; directly serves the metal priority | Fully pulled (5,562/5,562) after the first fetch stopped at 1,615. Cracks and scratches are split 80/15/5 at group level under v7. Still a limited steel corpus. | [Hugging Face dataset](https://huggingface.co/datasets/Zhaosxian/SteelDefectX) |
| Magnetic Tile | 1,295 images: 51 ceramic cracks, 323 non-crack defects, 921 clean | Factory-like ceramic/tile surfaces with distinct Crack and Free classes | Licence must be rechecked before release. Blowholes, breaks, frays and unevenness are hard negatives, not crack masks. | [Source repository](https://github.com/abin24/Magnetic-tile-defect-datasets.) |
| KolektorSDD1 | 380 images: 49 epoxy cracks, 331 clean | Real manufacturing imagery with paired masks; fills the epoxy category | Evaluation-only. The “plastic embedding” is epoxy resin, so it must not inflate general-plastic coverage. | [KolektorSDD project](https://www.vicos.si/resources/kolektorsdd/) |
| MVTec AD selected subset | 153 images: 37 tile/capsule crack masks, 116 scratch-labelled examples | Small independent crack/scratch evaluation signal spanning ceramic, wood, metal and plastic-like objects | Evaluation-only and CC BY-NC-SA 4.0. Scratch masks are anomaly blobs, not thin scratch labels. Capsule is pharmaceutical-domain data and is explicitly flagged. | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) |

### Mask acceptance decisions

- **CrackSeg9k `cracktree` is rejected.** Its measured mean, median, p10 and p90 widths are all exactly 2 px. This is a constant-width annotation artefact, not a physically meaningful crack-width distribution.
- **Rissbilder and Volker are plaster/render.** They are German facade images. Calling them concrete falsely inflated concrete coverage; their labels are retained but their material is corrected.
- **Wood `knot_with_crack` is not a crack target.** The annotation follows the whole knot region. 1,115 of 2,271 apparent positives were rounded blobs, which would teach region detection instead of crack segmentation.
- **SteelDefectX passes the thin-mask audit.** Median widths are approximately 4.1 px for steel cracking and 7.3 px for steel scratches, consistent with thin structures rather than anomaly blobs.
- **MVTec scratch masks are not segmentation labels.** Their median width is 33.6 px, versus roughly 4–15 px for accepted thin structures. They remain valuable as image-level scratch stress tests.

## 4a. Downloaded and staged for annotation: casting impellers, Rajkot

Kaggle `ravirajsinh45/real-life-industrial-dataset-of-casting-product`, CC-BY-NC-SA-4.0. 7,348 top-view images of stainless-steel submersible pump impellers produced by **Pilot Technocast, Rajkot, Gujarat**, captured on a **fixed camera at constant working distance under stable lighting**.

That capture geometry is why this source matters more than its size suggests. Every other real source in the corpus is handheld, web-scraped or shot on a lab bench; this is the only one photographed the way the deployment station photographs (architecture §5). It is also an actual Indian MSME foundry product, which no other source is.

Its labels are classification only — `ok_front` and `def_front`, no masks — and the two halves therefore cannot be treated the same way. `dataset/adapters.py::adapt_casting_impeller` splits them:

- **`ok_front` (3,137) → imported as negatives.** Establishing that an image is defect-free requires no annotation, and clean product under the deployment camera is exactly what the false-positive requirement needs most.
- **`def_front` (4,992) → not imported.** An image that contains a defect but carries no mask is neither a positive (there is nothing to segment against) nor a negative (training on it as clean teaches the model to ignore the very defects it exists to find). Filing it under either role would silently corrupt the corpus. These are copied to `data/label_queue/casting/` and enter only once masks exist.

This is the same rule already applied to SDNET2018, Özgenel and the KolektorSDD2 defective subset in §6. It is restated here because the temptation is strongest for this source: it is the best-matched data we have, and it would be easy to justify importing 4,992 defective images on the strength of that fit alone.

## 4b. Roboflow Universe imports, and the gates they had to pass

Fifteen Universe projects were reviewed. Four contributed pixel targets; the rest were rejected, and the rejection reasons are more informative than the acceptances. All credits and licences are in `docs/ATTRIBUTION.md`.

| Project | Kept | Material | What it added |
|---|---:|---|---|
| `gazxard/pipe-crack-detection` v1 | **937** | plastic | The plastic gap, closed. PVC pipe cracks under even lighting |
| `steel-reuse/steel-reuse-cracks-ysu8b` v1 | **293** | steel | Nearly doubles real steel crack coverage |
| `aleksandr-endoscope/endoscope-cracks-segmentation` v3 | **131** | steel | Borescope optics, **not** conveyor capture — flagged, not headline evidence |
| `heechan/cracked_phone` v4 | **12** | glass | The only real glass masks in the corpus |

These are COCO polygon exports, not PNG masks. Polygons are the better choice: they rasterise to the exact annotated boundary at whatever resolution we pick, with no resampling of an already-resampled label.

Roboflow augments *before* export, so many rows are transforms of one photograph, sharing the stem before `.rf.`. That stem is carried as the group key, so the split stage cannot put two augmentations of one photo on opposite sides. Without it these imports would have inflated every number they touch.

Three filters decide what enters, and roughly 15 % of candidate rows plus two entire sources failed them:

1. **Per-class trust.** A dataset being good as a whole does not make every class in it usable. `pipe-crack-detection` pairs a real `PVC pipe crack` class with `Paper crack` (957) and `Dummy crack` (954); importing all three would teach the model that paper and staged cracks are factory defects. Only the first is used.
2. **Whole-object rejection by measurement, not by eye.** Contact-sheet review of the first import found annotators who outlined the *entire phone* rather than the crack across it, and images where the mask covered the whole frame. At 30–100 % foreground those teach the model that "defect" means "most of the picture". Real thin defects in this corpus run 0.1–5 % of pixels, so anything above **15 % foreground** is dropped. 19 of `cracked_phone`'s 31 rows went this way.
3. **Blob rejection by shape.** A crack is elongated: its skeleton is long relative to its area, while a filled region of the same area has a short skeleton. Anything whose area-to-skeleton ratio exceeds **20 px** is dropped — the same rule that quarantined MVTec's 33.6 px scratch masks. This catches region outlines small enough to slip past the fraction test.

Two sources were disabled outright after their contact sheets: `instacash/mobile-minute-cracks` (the images are **line-art schematics of phones**, not photographs of a surface) and `yolo-r8fla/glass-detection-uymwf` (3 rows survive, all near-black and unverifiable). `dataset-rmy16/glass-dataset-scratch` looked like the answer to the glass gap until inspection showed its 1,443 polygons annotate the **glass panel region**, not the defect — the actual scratch annotations carry no polygons at all.

Four projects, including the most valuable one, could not be fetched: `shahin-workspace/metal-defects-jn5zf` has 1,098 instance-segmentation images with 781 crack annotations on **non-steel metal, our weakest material**, but has no generated version, so the API cannot export it. Forking it into our own workspace to generate a version is the obvious next move.

## 5. Source catalogue: clean backgrounds and hard negatives

| Source | Current contribution | Why it is included | Why it is not a crack-positive source | Reference |
|---|---|---|---|---|
| NEU-DET | 1,499 steel hard negatives | Real steel scratches, inclusions, patches, pitting and rolled-in scale train the model not to over-flag industrial texture | `crazing` is genuinely crack-like but supplies boxes/no usable pixel masks. Zero-mask positives would teach cracks as background, so that class is excluded completely. | [NEU surface-defect database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) |
| Severstal steel competition | 3,998 images: 2,078 hard negatives and 1,920 clean steel surfaces | Real rolled-steel texture and non-crack defects are excellent false-positive controls | Its labelled defects are scale, patches, inclusions and scratches—not cracks. | [Kaggle competition](https://www.kaggle.com/competitions/severstal-steel-defect-detection) |
| GC10-DET | 2,300 metal hard negatives | Weld lines, creases, pits, inclusions, oil/water spots and other real metal clutter represent practical distractors | It has no crack class. This is an intentional non-crack source, not missing supervision. | [GC10-DET project](https://github.com/littledeep/GC10-DET) |
| KolektorSDD2 | 2,979 clean industrial/plastic-surface images | Empty-mask images are high-value, safe plastic-class backgrounds | Defective images mix scratches, spots and possible cracks without separable labels. They are excluded rather than mislabeled as background. | [KolektorSDD2](https://go.vicos.si/kolektorsdd2) |
| SDNET2018 subset | 6,000 clean concrete images | Large clean pool selected only from the non-cracked folders | The cracked half is classification-only; without a boundary it cannot supervise segmentation. | [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) |
| Özgenel surface-crack subset | 5,830 clean concrete images | Adds clean texture diversity from its Negative folder | Classification-only; cracked images are excluded because their exact pixels are unknown. | [Kaggle dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |
| Describable Textures Dataset | 5,464 clean mixed textures | Broad texture variation suppresses texture-only crack responses | Research-only and not industrial evidence. The `cracked` category is explicitly excluded. | [DTD project](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |

## 6. Rejected and deferred data: why more images are not automatically better

| Candidate | Decision | Reason |
|---|---|---|
| Kaggle crack-segmentation re-upload | Rejected | It overlaps CrackSeg9k but lacks a unified annotation convention. It would add duplicates and reintroduce inconsistent mask thickness. |
| NEU `crazing` | Rejected, not made negative | Real crack networks without segmentation masks. Empty-mask training would be actively wrong. |
| SDNET2018 cracked half | Rejected as foreground | Image-level class labels are insufficient for pixel segmentation. |
| KolektorSDD2 defective subset | Rejected | Mixed defect semantics; some may be cracks. Treating them as background would silently poison the labels. |
| MVTec scratch masks | Rejected as pixel targets | Broad anomaly areas, not scratch boundaries/centrelines. |
| MVTec pill and hazelnut crack classes | Rejected | Pharmaceutical tablet and nut-shell surfaces are out of domain and would flatter industrial generalisation. |
| OmniCrack30k | Deferred | Potentially valuable, including steel, but access is author-gated and non-commercial licence terms must be reviewed. |
| Culvert-Sewer Defects | Deferred/out of scope | Useful for sewer CCTV but author-gated and not the current factory surface-photography problem. |
| Pipeline gamma-radiography data | Rejected | X-ray imagery is a different sensing modality from phone/webcam/borescope RGB captures. |
| Untraceable web/Hugging Face re-uploads | Rejected unless provenance is known | Many are repackaged CrackSeg9k; file count is not new independent evidence and may create train/test leakage. |

## 7. How generated data is made useful rather than misleading

Generated data is not a replacement for real factory images. Its role is controlled augmentation of morphology, texture, nuisance features and camera conditions between real-data collection rounds.

### 7.1 Photographic compositing: `data/synth`

`defectforge/generate.py` selects a verified crack-free photographic background, renders a procedural crack or crack-like distractor at 512×512, then downsamples to the 256×256 training size. Rendering above the final resolution produces anti-aliased thin structure that is closer to an optical capture than a one-pixel drawing.

Each generated sample has provenance: patch ID, generator version, deterministic seed, background ID/path, material, sample kind, sample index, foreground fraction and render/geometry parameters. The pair `(run_seed, sample_index)` regenerates the same image, so a synthetic run is reproducible and can be deleted/rebuilt after an engine change.

Procedural positives vary branching, taper, scale, contrast and material-aware geometry. Procedural negatives use the *same rendering pipeline* to create scratches, seams, grain/brushed-metal lines, grout, cable shadows and spatter, but their masks are empty. The model therefore cannot use rendering style as a shortcut; it must learn structural differences that reduce false positives.

### 7.2 Verified background policy

`defectforge/backgrounds.py` admits a background only if:

1. It is canonical `negative`, never a positive or hard negative.
2. Its existing mask is empty.
3. Its clean status comes from a source folder/label guarantee, not an automatic guess.
4. It belongs to `train` or the special `wood_bg` split, never val/test.
5. It has adequate brightness, contrast and unclipped dynamic range for a visible defect.

Crack-free crops from positive images are permitted only if both the crop and a safety margin around it have empty masks. This creates realistic material/camera substrates without leaking a labelled crack into the generated label.

An attempted black-hat/Frangi ridge filter was removed after testing: it could not reliably distinguish known cracks from legitimate industrial texture. An arbitrary threshold would have created false confidence, so the dataset relies on construction guarantees and contact-sheet review instead.

### 7.3 Companion DefectForge engine: from ablation-only to the sole scratch source

The companion DefectForge engine uses physically based height-field rendering and procedural textures. Its first adapted output, `data/df_patches`, held 45,627 patches (12,556 cracks, 33,071 hard negatives) and was kept as a comparison set only: merging the two synthetic *crack* domains reduced unseen-material performance in every ablation, even when in-domain and false-positive metrics improved.

The three-class reframe changed its status. Our compositor produces cracks and crack-like distractors but **no scratches**, and real scratches exist on exactly one material (steel). DefectForge is therefore the only synthetic scratch source available, so it was re-adapted as `data/df_patches2` with a scratch kind:

| set | crack | scratch | negative | total |
|---|---:|---:|---:|---:|
| `data/synth` (our compositor) | 96,953 | 0 | 50,000 | 146,953 |
| `data/df_patches2` (DefectForge) | 27,295 | **5,413** | 12,919 | 45,627 |

`data/synth` is itself the merge of two generation passes: `synth_core` (109,198) and `synth_fix` (38,646), the latter a ceramic/plastic re-generation at corrected crack width. Both are retained separately for provenance; only the merged directory is trained on.

**Gap between intent and behaviour, found and fixed at v8.** `experiments/run_3class.sh` documented that only the *scratch* kind was drawn from DefectForge, precisely to avoid re-opening the settled crack-mixing result. The code could not express that: `--synth-kinds` filtered the concatenated frame by kind, not per source, so passing `crack,scratch,negative` over both directories admitted DefectForge's 27,295 cracks as well — **30 % of every synthetic crack the model saw**. Every three-class measurement up to and including tag `V7` was taken under it.

`bench/data.py` now accepts per-source kinds, `--synth data/synth:crack|negative,data/df_patches2:scratch`, which is what the recipe always claimed to do. Because the ablation that argued against crack-mixing was measured under the *binary* head on splits v4, the fix is not assumed to help: run `V8B` is the old command on the same v7 splits, so the change is measured rather than asserted.

**Generation is complete and static.** The last synthetic patch was written on 2026-08-05 06:37 (`df_patches2`); no generator has run since. Corpus growth from here comes from real imagery and labelling, not from more rendering.

### 7.4 Device realism

`bench/camera_aug.py` applies geometric transforms to image and mask together, and photometric effects to the image only. It offers two profiles, selected by `--camera-profile`.

**`conveyor` (default from v5 onward)** models the deployment station: a fixed camera at constant working distance under controlled illumination, with the part moving along one axis. It is deliberately *narrower* than the handheld model, and spends the budget it saves on the two artefacts that actually dominate a line:

- **Belt-axis motion blur.** Handheld blur is smeared in whatever direction the hand moved, so the generic model randomises the angle over 180°. A part on a conveyor moves along one fixed axis under a fixed camera, so smear direction is a property of the installation. Randomising it would teach the model that defect orientation and blur orientation are independent, which on a real line they are not — a scratch parallel to travel and one across it degrade differently.
- **Specular highlights.** A blown-out reflection of the LED bar off polished steel or glazed ceramic is bright, thin and directional — the same description as a scratch. This is the single most important negative artefact for the material we prioritise, and it is photometric: it adds light, not a defect, so the mask must not follow it.

Also present: in-plane rotation over the full circle and small translation (a part's position and orientation on the belt are the things that genuinely vary), mild vignetting, overhead LED falloff, sensor noise, JPEG, and occasional defocus for part-height variation.

**Barrel distortion is deliberately absent from the conveyor profile.** A machine-vision lens on an inspection station is selected and calibrated to be rectilinear, so simulating fisheye would train the model to undo a distortion the deployment optics do not have.

**`handheld`** retains the previous model — perspective, barrel distortion, rolling-shutter shear, chromatic aberration, wide white-balance variation — and exists so that every benchmark measured before v5 remains reproducible, and so phone or borescope capture can be revisited without rewriting the augmentation. Real captures on those devices are still required before either is claimed as supported.

### 7.5 What a training run actually consumes

Two numbers are easy to confuse: how many patches exist on disk, and how many a run is allowed to draw from. They differ by a leakage filter, and the gap is large enough that quoting the disk figure would overstate the training set by more than a third.

Accounting for the v8 recipe (`--synth data/synth:crack|negative,data/df_patches2:scratch`):

| stage | patches |
|---|---:|
| on disk, both sources | 192,580 |
| after per-source kind selection (§7.3) — DefectForge cracks and negatives dropped | 152,366 |
| **dropped: background belongs to a held-out split** | **−52,416** |
| **admitted to training** | **99,950** |

The dropped 52,416 are the whole point of the mechanism. A synthetic patch inherits its background photograph's split, and if that photograph is in `test_factory` or `val`, the patch smuggles evaluation pixels into training under a brand-new ID that no downstream leakage check can recognise. The filter re-derives this from the *current* `manifest_split.csv` at load time rather than trusting `backgrounds.csv`, because splits get re-cut after backgrounds are generated — v7 is the third re-cut, and each one invalidates the previous mapping.

Composition of the 99,950 admitted patches:

| kind | patches | source |
|---|---:|---|
| crack | 62,326 | compositor only |
| negative | 32,211 | compositor only |
| **scratch** | **5,413** | DefectForge only — the sole synthetic scratch source |

Under the superseded v7 command the same accounting gave 140,164 patches, of which 89,621 were cracks including 27,295 from DefectForge. Any comparison between a `V7` and a `V8A` number is a comparison across two different synthetic corpora as well as two mixing ratios.

Synthetic data is not sampled uniformly against real data. `--synth-frac` is the probability that a training sample is drawn from the synthetic pool at all, and `--scratch-frac 0.35` then reserves 35 % of the *defect* draws — in both the real and synthetic branches — for scratch.

The scratch quota is load-bearing: at its natural ~6 % frequency the class was never predicted on a single pixel (recall 0.000) while cracks scored 0.53 clDice. Quota-sampling took scratch clDice to 0.88. The rebalancing is done at the sampler and deliberately **not** in the loss — loss weighting buys scratch recall by over-painting, and over-rejection is the failure mode this system exists to avoid.

**`--synth-frac` moved from 0.45 to 0.25 at v8.** 0.45 was set when the synthetic pool was the only signal several materials had: plastic held 0 real crack masks, steel 300. v7 changed that — 936 real plastic masks and 685 real steel — so a batch that is 45 % rendered is now spending its capacity on the weaker of two available signals. At 0.25 the real share of a batch rises from 55 % to 75 %. The synthetic pool remains 13× larger than the 7,666 real defect images in `train`, so this shifts emphasis; it does not remove the dependency.

Glass is the exception and must be stated as one: 12 real masks against 3,739 synthetic glass patches. Nothing about the glass class is evidence, at any mixing ratio.

## 8. Plan for the next generated dataset: factory-balanced `synth_factory_v1`

The objective is not simply to generate more images. It is to make a separately versioned training corpus that specifically closes factory-QC weaknesses without corrupting the proven baseline.

### Phase A — preserve the evidence base

1. Keep v4 validation and test images immutable.
2. Rebuild `data/backgrounds.csv` from only eligible `train`/`wood_bg` images after any split change.
3. Store every synthetic run in a new directory with its own provenance file; do not overwrite `data/synth`.
4. Generate and inspect contact sheets by material, class and device profile before training.

### Phase B — factory-balanced procedural synthesis

Create `data/synth_factory_v1` with the following design:

| Component | Plan | Why |
|---|---|---|
| Material sampling | Equal target counts for steel, ceramic, epoxy/plastic and mixed metal; bounded civil auxiliary sampling | Native data frequency would otherwise drown scarce factory materials. |
| Crack geometry | Fit width, length, branching and foreground fraction to real SteelDefectX, Magnetic Tile, KolektorSDD1 and accepted CrackSeg9k masks | Measured distributions are more defensible than arbitrary lines. |
| Scratch class | Generate a separate thin scratch mask, with bright/dark variants and material-aware orientation/texture | Provides valid supervision for the planned three-class model; MVTec blobs are not reused. |
| Hard negatives | Increase weld seams, brushed-metal grain, tool marks, pits, oil/water spots and grout | These are the false-positive modes that matter in factory inspection. |
| Device profiles | Explicit borescope, webcam and phone profiles with mask-safe geometry | Targets the devices specified in the requirements, not generic augmentation. |
| Synthetic ratio | Start at ≤45% of training samples per epoch; sweep only using frozen real tests | A 60% synthetic share previously collapsed unseen-material performance. |

No generated corpus becomes the recommended default unless a ≥3-seed comparison on v4 improves factory metrics without breaking the clean-surface false-positive-area limit of 0.5%.

## 9. Plan for more real images and SAM2-assisted labelling

The steel experiment provides the central lesson: adding 180 real steel crack images raised steel clDice from approximately 0.09–0.14 to 0.61–0.63. Real material breadth is therefore more valuable than another 100,000 synthetic patches.

### Targeted collection

Collect at least 50–100 confirmed defects and an equal or larger number of clean images for plastic, aluminium/copper/painted metal and pipe interiors. For each material, capture phone, webcam and borescope views under several lighting distances and include realistic distractors. Record part/batch/device IDs to preserve group-safe splitting. Reserve a final test portion **before** annotation/training and never reuse it as synthetic background.

### SAM2-assisted, human-verified workflow

1. Collect unlabelled real factory images; store source, device, material and part/batch ID.
2. Run SAM2 plus an independent thin-structure/edge proposal method.
3. Keep only thin, elongated candidates where SAM2 and edge evidence agree; never accept a mask from a single high-confidence model alone.
4. Review contact sheets showing source image, overlay and mask. A human marks each candidate accept, edit or reject and assigns crack/scratch.
5. Save reviewer, timestamp, method version, source device and edit status in a pseudo-label manifest.
6. Use only accepted/edited pseudo-labels in training. No pseudo-label enters validation or test.

This procedure uses automation to reduce annotation cost but retains human accountability where mask semantics matter.

## 10. Quality gates for every dataset update

```powershell
python dataset/index.py
python dataset/normalize.py
python dataset/split.py
python dataset/qa.py --strict
```

The update is rejected if QA detects group/phash leakage, a foreign unseen material, a missing evaluation split, invalid image/mask pairing, non-binary masks or crack pixels in `test_negatives`. Structural checks are followed by per-material and per-device contact-sheet review.

The machine-readable source registry is `dataset/sources.yaml`; downloaded source hashes and retrieval details are in `data/raw/.lock.json`. Licences and attribution obligations must be rechecked before any release beyond the hackathon, especially for Magnetic Tile, Kaggle-derived datasets, MVTec AD, KolektorSDD2 and any author-gated dataset.
