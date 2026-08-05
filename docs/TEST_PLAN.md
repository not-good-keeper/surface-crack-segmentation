# Acceptance Test Plan

Case IDs are the **Phase 2 §6.3** cases (`T-01…T-25`), unchanged. Cases added after
Phase 2 was written continue the series from `T-26`. Requirement IDs are Phase 1's
(`FR-01…FR-19`, `NFR-01…NFR-15`); `REQUIREMENTS.md` holds their status.

| Status | Meaning |
|---|---|
| **Automated** | Runs unattended and fails loudly; listed with the command |
| **Scripted** | A repository tool produces the evidence, but the pass criterion is read by a person rather than asserted |
| **Interface** | Belongs to the Phase 2 application layer, outside this repository |
| **Not implemented** | Defined, not built |
| **Manual** | Requires the physical rig |

---

## 1. Test data

Every case runs against one of these. None of them is generated at test time — a fixture
that regenerates itself makes the repeatability case meaningless (Phase 2 §6.4).

| Set | What it is | Where |
|---|---|---|
| Frozen factory split | 156 real images, sealed; opened only for final measurement. Manifest sha recorded | `test_factory` |
| Clean-surface set | 4,626 real defect-free images, never trained on | `test_negatives` |
| Held-out scratch set | 31 real steel scratches with thin masks | `test_factory_scratch` |
| Unseen-material set | 1,156 wood images, no wood positive ever in training | `test_unseen_material` |
| Blob-convention set | 100 MVTec scratches whose masks outline the anomaly region, not the scratch | `test_scratch_blob` |
| Synthetic shapes | Lines and arcs of known area, length and width, including a 45° diagonal | `app/test_pipeline.py` |
| Score-map fixtures | Hand-built logit arrays where argmax and the threshold rule disagree, and where all three classes sit near 0.33 | `app/test_pipeline.py` |
| Unreadable files | Truncated PNG, zero-byte file, oversized image, text file with a `.png` extension | `app/test_pipeline.py` |
| Reference targets | Printed lines of known length and width, for the millimetre conversion | physical rig |

---

## 2. Test cases

### Pipeline and contract

| ID | What it does | Pass criterion | Status |
|---|---|---|---|
| T-01 | Run 200 valid images through `Inspector` | All produce a record; no unhandled exception; station ID, product ID and timestamp on each | **Scripted** — `app/batch.py` over a directory |
| T-02 | Compare deployed preprocessing against the training loader element by element on 50 images | Tensors match within float tolerance. Any difference in resize, scaling or normalisation fails | **Not implemented** — the highest-value missing case, see §5 |
| T-03 | Inspect the ONNX graph: names, shapes, dtype, output bounds | Input `input` (1,3,256,256) float32; output `logits`; values outside [0,1] present, confirming raw logits | **Automated** — `bench/export_onnx.py::check_parity` |
| T-04 | Collect every value in the class map over the frozen split | Only 0, 1 and 2 appear | **Automated** — `test_classes_are_mutually_exclusive` |
| T-05 | Feed score-map fixtures where argmax and the threshold rule disagree | Below both floors is background; where both pass, the higher score wins. A 0.34 / 0.33 / 0.33 pixel is background, not a crack | **Automated** — `test_classes_are_mutually_exclusive`, and `bench/class_thresh.py` measures the effect |
| T-06 | Change `Profile.crack_thresh` and re-run pipeline and interface | Both reflect the new value with no other edit; no threshold hard-coded anywhere | **Automated (pipeline half)** — the CLI override was removed so `Profile` is the only source; interface half is Interface |

### Defect handling and geometry

| ID | What it does | Pass criterion | Status |
|---|---|---|---|
| T-07 | Run the clean-surface set | ≥98 % return an empty region list; false-positive area ≤0.5 % | **Automated** — `bench/final_eval.py`, uncapped. Measured **0.33 %** |
| T-08 | Submit a clean image and each unreadable file | Clean is `status=clean` with an empty region list; each bad file is an acquisition failure. Neither is ever reported as the other | **Automated** — `test_clean_product_is_explicit`; `app/batch.py` records `acquisition_failure` |
| T-09 | Measure the synthetic shapes, including a 45° line | Length within ±3 %, maximum width within ±1 px. The diagonal is not understated, confirming the √2 step | **Automated** — `test_geometry_is_correct_by_construction` |
| T-10 | Measure a thin crack with a local bulge | The reported maximum width is the *inscribable* width. The behaviour is recorded and the interface note is present | **Scripted** — documented in `INTEGRATION.md`; not asserted |
| T-11 | Build regions of 20 px and 30 px area, and 4 and 8 skeleton pixels | Regions below either floor are dropped, those above kept | **Automated** — `test_min_area_filter` |
| T-12 | A fixture where a crack and a scratch touch | Two separate regions, one per class, each with its own geometry | **Automated** — `test_touching_classes_stay_separate` |
| T-13 | Compare overlay and class-map dimensions with the source | Both at source resolution, not 256×256; overlay registers to the source | **Automated** — `test_overlay_survives_rescaling` |

### Records, batch and reproducibility

| ID | What it does | Pass criterion | Status |
|---|---|---|---|
| T-14 | Read every field of 100 stored records | Image SHA-256, model version and artefact hash, profile version, station and product ID present on all | **Partial** — `build_record` writes them; the 100-record sweep is not asserted |
| T-15 | Run the same 100 images twice with model and profile fixed | Identical class maps, identical geometry, identical fields apart from the timestamp | **Automated** — `test_determinism` |
| T-16 | Batch 500 images, export CSV and JSON, recalculate totals independently | One row per region plus one per clean image; failures carry a status; totals match | **Scripted** — `app/batch.py`; independent recalculation not asserted |
| T-17 | Query history on each filter separately and combined | Each result set matches an independently computed set, including when empty | **Interface** — needs the SQLite store |

### Model, performance and governance

| ID | What it does | Pass criterion | Status |
|---|---|---|---|
| T-18 | Evaluate the frozen split per class and per material | clDice, tolerant F1 and IoU per material; detection and typing reported separately, never merged | **Automated** — `bench/final_eval.py`, `bench/per_material.py` |
| T-19 | Model size and per-image latency over 500 runs on one CPU thread, then end to end | 5.8 MB float32; inference median ≤30 ms; end-to-end recorded with thread count. No ARM figure quoted | **Automated (inference)** — `bench/export_onnx.py`. End-to-end stage timings **not implemented** |
| T-20 | Disable the network, run a live inspection and a batch run, trace syscalls | Both finish; no socket opened | **Not implemented** — the import audit exists, the syscall trace does not |
| T-21 | Search every rendered page for a percentage, a score, or accept/reject wording | No probability, confidence or verdict wording appears anywhere | **Interface** |
| T-22 | Compare the coverage table on screen against the metrics file | Every number comes from the metrics file; none hard-coded | **Interface** — `bench/per_material.py` produces the file |
| T-23 | Replace the `.onnx` with a different file of the same name and start | Hash mismatch detected at load; inspection does not start; status names the problem | **Not implemented** — the hash is recorded but not verified at load |
| T-24 | Put synthetic and unreviewed samples into validation and test, then run the QA gate | The gate fails and names the offending files; no near-duplicate crosses a split boundary | **Automated** — `dataset/qa.py --strict`; `bench/data.py` raises on synthetic in a non-train split |
| T-25 | On the rig: photograph reference targets, then a timed walkthrough with five users | Length within ±5 %, width within ±1 px; review in ≤3 interactions | **Manual** |

### Added after Phase 2

| ID | What it does | Pass criterion | Status |
|---|---|---|---|
| T-26 | ONNX parity against the torch model | Max abs logit drift <1e-3 and per-pixel argmax agreement >0.999. A segmentation head is more export-sensitive than a classifier — every pixel is its own argmax, so a small shift moves the decision on exactly the thin structures the system exists to find | **Automated** — `bench/export_onnx.py`, blocks the export on failure |
| T-27 | All-background baseline reported beside every result | A model predicting nothing scores 95.6 % pixel accuracy on `test_factory`. If the metric set cannot separate that from a real model, the metric set is wrong | **Automated** — `bench/metrics.py::all_background_baseline` |
| T-28 | Input transform selected on `val`, audited against NFR-03 | `bench/prep_sweep.py` cannot be pointed at a test split. Any candidate must report `fp_area` on `test_negatives`: the best-headline transform (`median3`, clDice +0.016) breaches the 0.5 % ceiling at 0.61 % and is rejected on that basis | **Automated** — selection split is a module constant, not a flag |
| T-29 | Deployment applies the transform the weights were trained with | The spec is stamped into ONNX `metadata_props` and read back by `app/inference.py`. Skipping it does not raise — it silently costs detection 0.962 → 0.949 | **Automated** — `export_onnx.py::stamp`, `Inspector.__init__` |
| T-30 | Per-epoch history stored for every run | Best-epoch selection over a long run is a lottery unless the tail is stable; the history is what makes that auditable (ADR-019) | **Automated** — every `data/bench/*.json` carries `history` |

---

## 3. Traceability

Phase 2 §6.5, updated for the cases above and the withdrawals in `REQUIREMENTS.md`.

| Requirement | Covered by | Requirement | Covered by |
|---|---|---|---|
| FR-01 | T-01, T-08 | NFR-01 | T-19 |
| FR-02 | T-08 *(partial — no focus/exposure gate)* | NFR-02 | T-19 |
| FR-03 | T-02 *(not implemented)* | NFR-03 | T-07 |
| FR-04 | T-13 | NFR-04 | T-18 *(partial — no per-material profiles)* |
| FR-05 | T-04, T-05, T-12 | NFR-05 | T-20 *(not implemented)* |
| FR-06 | T-07 | NFR-06 | T-15 |
| FR-07 | T-09, T-10, T-11, T-25 | NFR-07 | T-24 |
| FR-08 | T-13 | NFR-08 | T-18 |
| FR-09 | T-16 | NFR-09 | T-21 |
| FR-10 | T-14 | NFR-10 | T-08 *(partial)* |
| FR-11, FR-12 | T-24 | NFR-12 | T-18 |
| FR-13 | T-22 | NFR-13 | T-23 *(not implemented)* |
| FR-17 | T-25 | NFR-14 | T-25 |
| FR-18 | T-17 | NFR-15 | T-14 |
| FR-19 | T-16 | Model contract (Phase 2 §3.4) | T-02, T-03, T-06, T-26, T-29 |

FR-14 has no covering case because it is not implemented. FR-15, FR-16 and NFR-11 are
withdrawn and have none by design.

---

## 4. Running the suite

```bash
# 1. dataset must regenerate from source and pass strict QA before any training claim
./.venv/Scripts/python.exe dataset/split.py
./.venv/Scripts/python.exe dataset/qa.py --strict          # T-24

# 2. pipeline unit and integration cases
./.venv/Scripts/python.exe app/test_pipeline.py            # T-04,05,08,09,11,12,13,15

# 3. full uncapped evaluation of one checkpoint, both decision rules
./.venv/Scripts/python.exe bench/final_eval.py --tag V10_22   # T-07, T-18, T-27
./.venv/Scripts/python.exe bench/per_material.py --tag V10_22 # T-18

# 4. export, parity and latency — idle machine only
./.venv/Scripts/python.exe bench/export_onnx.py \
  --models smpslim_timm-mobilenetv3_small_100 --classes 3 --tag v10 \
  --prep bilateral --weights data/bench/<checkpoint>.pt      # T-03, T-19, T-26, T-29
```

`--classes 3` and `--tag` are both mandatory on export: a three-class export written over
the binary one would leave every stored size and latency figure describing a model that
no longer sits at that path.

Latency is measured on an idle machine or not at all. The same model measured 96 ms,
103 ms and 412 ms during this project depending on what else was running, so
`export_onnx.py` refuses to report a number while other heavy processes are alive.

---

## 5. Automation status, honestly

Phase 2 §6.4 stated that 24 of 25 cases would run in GitHub Actions on every merge.
That is the target, not the current state.

| | Cases |
|---|---|
| Automated and failing loudly | 13 — T-03, 04, 05, 06(pipeline), 07, 08, 09, 11, 12, 13, 15, 18, 24, plus T-26…T-30 |
| Scripted, read by a person | 4 — T-01, T-10, T-16, T-19(end-to-end) |
| Interface, outside this repository | 5 — T-17, T-21, T-22, and the interface halves of T-06 and T-13 |
| Not implemented | 4 — **T-02**, T-14 (sweep), T-20, T-23 |
| Manual | 1 — T-25 |

**T-02 is the most valuable missing case.** It compares deployed preprocessing against
the training loader element by element. Every other case can pass while preprocessing
drifts, because both sides of every metric shift together — the failure is invisible to
the entire benchmark. It became more important, not less, once an input transform joined
the contract (T-29): there are now two ways for deployment to diverge from training
rather than one. It is cheap to write and should be the next test added.

There is no CI workflow in this repository yet. The commands above are the suite.

---

## 6. Defect handling

Phase 2 §6.6.

| Severity | Meaning | Response |
|---|---|---|
| S1 | A failure that makes output wrong without anyone noticing — preprocessing drift, the class rule not applied, clean confused with failed-to-read | Stop other work, fix, re-run the whole group |
| S2 | A test fails, or a requirement works only under restricted conditions | Fix within the current work package, or write the restriction into `REQUIREMENTS.md` |
| S3 | Cosmetic or usability issue with no effect on output | Backlog |

When a bug is found, a test that fails before the fix and passes after it is added and
kept. Golden fixtures are regenerated only by a deliberate committed change.

---

## 7. Reporting rules

Each exists because it was broken at least once during development.

1. **Single-seed numbers are not evidence.** Cross-material clDice measured 0.579 / 0.309 / 0.075 across three seeds of an otherwise identical configuration. Report mean ± std, with the split version and sha.
2. **Quote the sample size below ~50 images.** `test_factory_scratch` is 31 images; the glass subset of `test_factory` is 2. A clDice on the latter is noise with a decimal point.
3. **Class-agnostic geometry must never stand in for class accuracy.** Headline clDice and IoU are computed on `any_defect`. A model can score 0.72 clDice while typing more than half of crack pixels as scratch — which is the measured state. Read `crack_class_recall` beside them.
4. **Never merge detection and typing into one number** (NFR-08, T-18).
5. **`test_scratch_blob` reports detection and class only.** Its masks outline the anomaly region at 33.6 px median width against 7.3 px for a real scratch. Geometry against it measures agreement with a labelling convention this project rejected on a width audit, so IoU and clDice are withheld rather than reported low.
