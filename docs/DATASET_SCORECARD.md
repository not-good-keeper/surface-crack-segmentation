# Dataset Scorecard — "Completeness of datasets to test solution" (20 %)

Self-assessment against rubric item 5, with the evidence for each claim and the gaps
stated as plainly as the strengths. Every figure here is read from
`data/manifest_split.csv`, `data/report/normalize.txt` or a provenance parquet, and can
be regenerated. Nothing is quoted from memory.

Companion to `DATASET.md`, which is the full strategy document. This file answers one
narrower question: **is the data complete enough to test the solution, and how do we
know?**

---

## 1. Self-rating: 15 / 20

| Dimension | Weight | Score | One-line justification |
|---|---|---|---|
| Test-set coverage of the deployment case | high | 3 / 5 | 7 frozen eval splits, 6,677 test images — but only **156** factory-material crack images, the headline number |
| Negative / false-positive testing | high | 5 / 5 | 4,626 negatives incl. 1,902 hard negatives, scored with no subsampling |
| Leakage control | high | 5 / 5 | Group-wise splits with asserts that **caught two real leaks** |
| Rejection discipline | medium | 5 / 5 | Every exclusion measured and logged, not asserted |
| Class coverage (scratch) | high | 2 / 5 | 724 real scratches, **one material** (steel). Every non-steel scratch claim is transfer |
| Provenance & licensing | medium | 4 / 5 | 18 sources catalogued with licence + tier; some sha256 still `null` |
| Primary capture | medium | 0 / 5 | **None delivered.** No conveyor line, no Indian factory capture |

**Why not higher:** the headline test split is 156 images, real scratches exist on
exactly one material, and no image in this corpus was captured on the conveyor line the
system is designed for. Those are three real limits on how much a good score here can
tell you.

**Why not lower:** the evaluation suite is frozen, reproducible, leakage-tested by
assertions that have caught actual bugs, and large enough on the negative side that the
false-positive claim is not a sampling artefact.

---

## 2. What exists

### 2.1 Corpus after cleaning

`data/report/normalize.txt`: **53,236 rows in → 51,504 kept** across 18 real sources.

| role | n |
|---|---:|
| negative | 28,753 |
| hard_negative | 11,531 |
| positive (crack) | 10,496 |
| scratch | 724 |

### 2.2 The frozen evaluation suite

This is the part the rubric asks about. **6,677 test images in 7 splits**, plus 7,144
validation images, none of which share a parent image with training.

| split | n | what it is for |
|---|---:|---|
| `test_factory` | 156 | **headline.** Cracks on factory materials (steel, ceramic, plastic, epoxy) |
| `test_factory_scratch` | 31 | held-out steel scratches, geometry scored |
| `test_scratch_blob` | 100 | MVTec scratches — **detection and class only**, geometry deliberately withheld |
| `test_seen` | 379 | cracks on materials present in training — upper bound |
| `test_unseen_material` | 1,156 | wood. Transfer evidence, never headline |
| `test_negatives` | 4,626 | 2,724 plain + **1,902 hard negatives** |
| `val_unseen_material` | 229 | unseen-material *validation*, so early stopping is not contaminated |
| `val` | 7,144 | model selection |

Two design choices in that table are worth stating explicitly:

- **`test_scratch_blob` reports detection but not IoU/clDice.** A width audit found its
  masks have a 33.6 px median width against 4–15 px for every accepted thin-structure
  source. Scoring geometry against a mask convention we rejected would measure agreement
  with the wrong convention. Detection and class survive; geometry does not.
- **`val_unseen_material` exists separately from `test_unseen_material`.** Selecting an
  epoch on the same material you report transfer on is not transfer measurement.

### 2.3 Negative testing at full scale

`bench/final_eval.py` runs **every** negative, no subsampling — training caps eval at 40
batches, which would have rested the NFR-03 claim on a quarter of them. Measured on
V12_22 over all 4,626: `fp_area` **0.00135** against a 0.005 ceiling.

An all-background baseline is reported beside every result: it scores **97.91 % pixel
accuracy** on `test_factory` with clDice 0.0000 and detection 0.0000. That number is in
the output precisely so pixel accuracy cannot be quoted as success.

### 2.4 Synthetic data, and its declared limits

| source | n | composition |
|---|---:|---|
| `data/synth` | 146,953 | 96,953 crack + 50,000 negative |
| `data/df_patches2` | 45,627 | 27,295 crack + 12,919 negative + **5,413 scratch** |

Used at `--synth-frac 0.15` and **train-only** — no synthetic image appears in any
evaluation split. Synthetic patches whose background image belongs to a held-out split
are dropped at load time, which is checked in code rather than assumed.

---

## 3. Rejected data, and the measured reason for each

Rejection here is a measurement, not a judgement call. This is the section that most
directly answers "completeness": the corpus is smaller than it could be **on purpose**.

| rejected | n | measured reason |
|---|---:|---|
| `cracktree` | 175 | Width std/mean = **0.0001** — a constant-width skeleton tracing, not a segmentation mask. Training on it teaches a fixed stroke width |
| near-duplicates | 1,557 | Identical perceptual hash. Duplicates across a split boundary are leakage; within a split they are silent reweighting |
| NEU `crazing` | ≈300 | Real crack networks with **no masks**. Unusable as positives, and actively harmful as negatives — they would teach the model that visible cracks are background |
| KolektorSDD2 defective | all with GT foreground | Labels mix scratches, spots and real cracks with nothing to separate them. The clean surfaces are imported as negatives; the defective ones are not usable for a crack/scratch head |
| wood `cracked` | (excluded at ingest) | Real cracks, no masks — same reason as NEU crazing |
| MVTec scratch geometry | 100 kept, geometry dropped | 33.6 px median mask width vs 4–15 px for accepted sources |

**Sources rejected before download**, with reasons recorded so the decision is auditable:
Tianchi aluminium (registration gated on a CN mobile number) · injection-moulded
polypropylene segmentation set (**embargoed until end-2027**) · Culvert-Sewer (author
gated) · OmniCrack30k (licence gated).

The `crazing` and `cracked` exclusions are the ones worth defending: both are large,
free, on-topic image sets. Including them as negatives would have grown the corpus and
degraded the model, and nothing in a headline metric would have shown why.

---

## 4. Quality gates that run before anything trains

`dataset/qa.py --strict` exits non-zero, so it can gate a pipeline. It asserts:

1. No `group` spans train and any eval split — **group-wise**, so crops of one parent
   image cannot straddle the boundary.
2. `test_unseen_material` contains only the declared unseen material.
3. The unseen material does not appear in train.
4. The unseen-**validation** material is absent from train, val and `test_seen`.
5. No duplicate names in the manifest.
6. No row in `test_negatives` has `crack_px > 0`.
7. Every declared eval split is non-empty.
8. **No split exists that `EVAL_SPLITS` does not list.**
9. Real rows exist in train for both `positive` and `scratch`.
10. Negatives span every material the system claims to support.
11. Masks are strictly binary and image/mask pairing is intact (400-row sample).

Gates 4 and 8 are not hypothetical. **Gate 4 was written after a 1,492-image leak passed
as green.** **Gate 8 was written after the scratch and steel holdouts turned out never to
have been leakage-checked at all** — they existed in the manifest but were absent from
`EVAL_SPLITS`, so every check silently skipped them. A suite that only tests what it
remembers to list will always report clean.

Structural asserts cannot catch a mask that is correctly formatted and points at the
wrong thing, so contact sheets are rendered per source and per class and reviewed by eye
before any adapter is trusted. That check caught the wood knot-blob and NEU crazing
errors.

---

## 5. Where the data is genuinely incomplete

Stated here rather than buried, because a completeness score that omits them is worth
nothing.

| gap | evidence | consequence |
|---|---|---|
| **Scratch is one material** | 724 real scratches, 499 in train, all `sdx` steel | Every scratch claim on ceramic, plastic or glass is transfer, not measurement |
| **Headline split is small** | `test_factory` n = **156** | A 0.03 clDice difference is within sampling noise at this size |
| **Thin factory coverage** | ceramic 168 · plastic 935 · epoxy 49 · glass 12 | Per-material results on ceramic and below are indicative only |
| **No primary capture** | 0 images | Nothing in the corpus was shot on a conveyor under fixed lighting. Camera realism is *simulated* by `camera_aug`, not sampled |
| **Unseen-material transfer is weak** | wood: detection **0.102**, clDice 0.043 | Whole-frame resize transfers to an unseen material far worse than defect-centred crops did. Reported as a regression, not omitted |
| **Casting impellers unannotated** | 2,779 imported as negatives; `def_front` staged in `data/label_queue/` | The best-matched source available — pump impellers from a real fixed-camera line at Pilot Technocast, Rajkot — carries classification labels only. Its clean frames are used; its defective frames wait for pixel masks |

The wood result is the honest headline of this section. It got *worse* than the earlier
crop-trained model (0.411 → 0.102 detection), because the earlier number was measured on
defect-centred crops that handed the model a guaranteed centred defect. The regression is
real; so was the old number's flattery.

---

## 6. Reproducibility

- `dataset/split.py` regenerates `data/manifest_split.csv` deterministically from source.
  This was a defect until recently: no committed script produced `test_scratch` or
  `test_steel`, so the manifest could not be rebuilt.
- `dataset/sources.yaml` pins each source with URL, licence, tier and commercial-use flag.
- Splits are frozen and versioned (v7 current, v5 retained for comparison), so a result
  always names the split freeze it was measured under.
- `data/report/normalize.txt` is the audit trail for every drop, with the measurement
  that justified it.

**Known reproducibility gap:** several `sha256` fields in `sources.yaml` are still
`null`, recorded as "unknown, fill on first download". Until those are filled, a source
could change upstream without the pipeline noticing.
