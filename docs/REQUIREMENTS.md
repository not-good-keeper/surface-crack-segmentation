# Requirements — status against the submitted reports

Requirement IDs in this repository are the ones defined in **Phase 1 §7 and §8**
(`FR-01…FR-19`, `NFR-01…NFR-15`). They are not renumbered here. A document that
invents its own IDs forces a reader holding the submitted report to build the mapping
themselves, and any row they cannot map looks like a claim with no requirement behind it.

Test IDs are the **Phase 2 §6.3** cases (`T-01…T-25`); see `TEST_PLAN.md`.

| Status | Meaning |
|---|---|
| **Met** | Implemented in this repository and covered by a test that runs |
| **Partial** | Implemented in part; the gap is stated in the row |
| **Interface** | Belongs to the Phase 2 application layer, which is a separate deliverable from this repository |
| **Withdrawn** | Formally dropped, with the reason. Phase 2 §2 withdrew the first four; the rest are recorded here |
| **Not met** | In scope, not delivered |

---

## 1. Scope of this repository

This repository is the **inference pipeline and its evidence base**: dataset
construction and governance, training, evaluation, post-processing, the ONNX artefact,
and the batch entry point. The operator interface, the SQLite store and the HTTP
service described in Phase 2 §3 and §4 are a separate deliverable and are marked
*Interface* below.

The deployment assumption is a **fixed industrial camera above a conveyor** at constant
working distance under controlled illumination (Phase 1 §12.5). This narrows Phase 1's
original three-device intent, and the cost is explicit: factories that cannot mount a
station are excluded, and the system inspects what passes under the camera rather than
whatever an operator points at. Controlled geometry is what makes a 1–3 px defect
measurable at all, so the trade was judged worth making — but it is a narrowing.

---

## 2. Functional requirements

| ID | Requirement (Phase 1 §7) | Status | Where | Test |
|---|---|---|---|---|
| FR-01 | Accept PNG/JPEG frames from the fixed camera, and stored directories for batch runs | **Met** | `app/batch.py`, `app/inference.py` | T-01, T-08 |
| FR-02 | Validate format, focus, exposure and field of view before inference | **Partial** | Decode failure is caught and recorded as `acquisition_failure`; there is **no focus or exposure gate** | T-08 |
| FR-03 | Normalise valid input to the fixed 256×256 model input while preserving source identity | **Met** | `app/inference.py::preprocess`, matching `bench/data.py` exactly | T-02 |
| FR-04 | Produce a pixel-level mask for every predicted defect region | **Met** | `app/postprocess.py::class_map` | T-13 |
| FR-05 | Assign every foreground pixel one of background, crack, scratch | **Met** | 3-class softmax head; no other value can be emitted | T-04, T-05, T-12 |
| FR-06 | Return an empty result on a clean surface rather than forcing a prediction | **Met** | `build_record` writes `"empty": true` explicitly | T-07 |
| FR-07 | Report area, centreline length and maximum width per region, in physical units where calibration exists | **Partial** | Pixels only. Millimetre conversion needs a per-station scale reference that has not been captured | T-09, T-10, T-11 |
| FR-08 | Write an overlay image showing every region and its class | **Met** | `app/postprocess.py::overlay`, at **source** resolution | T-13 |
| FR-09 | Process a directory and produce a batch CSV report | **Met** | `app/batch.py` → `report.csv`, one row per region plus one per clean image | T-16 |
| FR-10 | Log image SHA-256, model version, processing configuration and output for every inspection | **Met** | `build_record` + `audit.jsonl` | T-14 |
| FR-11 | Require human verification before assisted or pseudo-labelled samples enter training | **Met by construction** | No pseudo-labelled sample is in the corpus. The labelling queue was cancelled; `data/label_queue` is unused | T-24 |
| FR-12 | Keep synthetic and pseudo-labelled samples out of validation and test | **Met** | `bench/data.py` raises if `synth_dir` is passed with `train=False`; `dataset/qa.py --strict` asserts it | T-24 |
| FR-13 | Display validation status and known limitations of each material profile | **Interface** | Measured coverage is produced by `bench/per_material.py`; rendering it is Phase 2 §4.7 | T-22 |
| FR-14 | Allow the operator to select the active material profile | **Not met** | One `Profile` for all materials. Per-material thresholds are designed (NFR-04) but not implemented | — |
| FR-15 | Generate an automated accept / reject / review-required decision | **Withdrawn** | Phase 2 §2. The system reports evidence, not a verdict: acceptance tolerances are per-customer and were never available | — |
| FR-16 | Permit an authorised inspector to override a decision with a mandatory reason | **Withdrawn** | Follows FR-15. With no verdict there is nothing to override, and with no accounts an action cannot be attributed | — |
| FR-17 | Provide an inspection dashboard | **Interface** | Phase 2 §4.3 | T-25 |
| FR-18 | Store inspection history and permit filtering on six fields | **Interface** | Needs the SQLite store of Phase 2 §3.7; this repository writes JSONL and CSV | T-17 |
| FR-19 | Generate summary statistics and downloadable reports for a batch or period | **Partial** | `app/batch.py` emits per-run totals and CSV; period queries need FR-18 | T-16 |

### Withdrawn, with the reason

**FR-15 / FR-16 — the verdict and the override.** Phase 1 specified accept / reject /
review-required driven by a derived *fracture* state. Both were withdrawn in Phase 2 §2.
The fracture state existed only to feed the verdict; the model outputs three classes and
that is what is reported. The consequence is recorded honestly rather than hidden: with
no verdict there is no override, with no override there are no accounts, and **the system
therefore cannot attribute an action to a person.** If a customer needs attribution,
accounts and the verdict come back together.

---

## 3. Non-functional requirements

| ID | Target (Phase 1 §8) | Status | Measured |
|---|---|---|---|
| NFR-01 | ≤6 MB float32; ≤2 MB int8 | **Partial** | 5.8 MB float32, 1.43 M params — inside the float32 limit. **int8 not delivered**, so the 2 MB target is unmet (Phase 2 §2) |
| NFR-02 | ≤60 ms median, ≤90 ms p95, single-thread CPU at 256×256 | **Met** | 26 ms median on one desktop thread. The phone/ARM figure is **withdrawn** — no ARM device was ever measured |
| NFR-03 | False-positive area ≤0.5 % on clean images | **Met** | **0.33 %** on `test_negatives` (4,626 clean images, uncapped). Also report `fp_image_rate` 7.5 % — the share of clean images with at least one spurious region, which is what an operator actually experiences |
| NFR-04 | One shared backbone and weight set, with material-specific threshold profiles | **Partial** | One backbone and one weight set across all materials — met. **Material-specific profiles are not implemented**; a single `Profile` is used, so FR-14 has nothing to select |
| NFR-05 | No network call at inference time | **Met** | `onnxruntime` on CPU, local weights, no client library in the path | 
| NFR-06 | Fixed model and configuration produce identical output | **Met** | `app/test_pipeline.py::test_determinism` |
| NFR-07 | Frozen, real-image, leakage-tested validation and test splits | **Met** | Splits v7, sha `c0fde17c96749567`, 51,504 images; `dataset/qa.py --strict` passes |
| NFR-08 | Report clDice, tolerant F1 and IoU per class | **Met** | `bench/metrics.py`; every stored run records all three plus per-class recall |
| NFR-09 | Overlay and region geometry visible to the operator | **Met** (pipeline) | Overlay and region list produced; display is Interface |
| NFR-10 | Invalid or weak-domain input degrades to recapture or review rather than silent acceptance | **Partial** | Unreadable input is an explicit `acquisition_failure`. **Weak-domain input is not detected** — an unsupported material produces a normal-looking result |
| NFR-11 | No automatic acceptance when preconditions are invalid | **Withdrawn** | Moot: there is no automatic acceptance to prevent (FR-15) |
| NFR-12 | Performance reported separately per material | **Met** | `bench/per_material.py`; ADR-016 makes it a reporting requirement, not a diagnostic |
| NFR-13 | Camera, model and storage health exposed | **Interface** | Phase 2 §4.8 |
| NFR-14 | A trained operator completes review in a small, fixed number of interactions | **Interface** | Phase 2 §4.3 |
| NFR-15 | Decision, override, threshold and model changes are traceable | **Partial** | Model version, image hash, profile and thresholds are recorded per inspection. Decision and override do not exist (FR-15/16); threshold changes are versioned in code, not in a store |

---

## 4. Changes since Phase 1

Phase 2 §2 recorded four. Two more followed from measurements taken after it was written.

| Change | Why | Recorded |
|---|---|---|
| Derived fracture state withdrawn | It existed only to feed the verdict | Phase 2 §2 |
| Phone/ARM latency target withdrawn | Never measured on ARM; carrying it would be an untested claim | Phase 2 §2 |
| int8 quantisation not delivered | float32 export is 5.8 MB, inside the 6 MB limit; the 2 MB int8 target is unmet | Phase 2 §2 |
| SQLite added as the queryable store | FR-18 needs filtering on six fields and FR-19 needs totals that reconcile | Phase 2 §2 |
| **Multi-device capture withdrawn** | The corpus holds almost no real borescope, webcam or phone captures. The requirement could only ever have been asserted against synthetic distortion, which tests the augmentation rather than the model | ADR-011 |
| **An input transform is part of the model contract** | A bilateral filter is applied before normalisation, at training and inference alike, and is stamped into the ONNX metadata. Skipping it silently costs detection | ADR-018, ARCHITECTURE §5.1 |

---

## 5. Input assumptions and constraints

- The surface fills most of the frame, is reasonably focused, and has one dominant material.
- Input normalises to 256×256 by **plain resize, not letterbox**. Matching the training loader matters more than preserving aspect ratio.
- The camera is fixed, rectilinear and calibrated. Barrel distortion is deliberately **not** modelled: simulating fisheye would train the model to undo a distortion the deployment optics do not have (ADR-011).
- Modelled station artefacts are belt-axis motion blur, specular highlights off polished steel or glazed ceramic, ring-light falloff, sensor noise and JPEG — `bench/camera_aug.py --camera-profile conveyor`.
- Confidence must not be displayed as a probability. The scores are uncalibrated (NFR-09, Phase 2 §4.1).

### Material coverage — the binding constraint on every claim

| Material | Real crack masks | Class typing | Position |
|---|---:|---:|---|
| Steel | 1,047 | 0.702 | Supported. Weakest geometry (clDice 0.529) — steel cracks are the thinnest in the corpus at ~4 px |
| Plastic | 936 | — | **One product only.** All masks are PVC pipe; a moulded casing is not covered |
| Ceramic | 120 | 0.315 | Thin coverage. Detection usable, typing weak |
| Epoxy | **0** | 0.279 | Detection works; the type label is transfer from other materials |
| Glass | 12 | — | **Not supported.** Not a measurement at any mixing ratio |
| Non-steel metal | **0** | — | **Not supported.** No masks in the corpus |

Scratch evidence is 624 of 724 masks on steel, so a `scratch` label on ceramic, plastic
or glass is transfer, not evidence.

**Crack/scratch typing is the open defect.** Detection is ~96 %; the correct *type* is
assigned to roughly half of crack pixels, biased toward scratch. Class recall tracks
per-material real training mass almost monotonically, so this is a data gap rather than
an architecture or threshold problem (ADR-016). The mask is therefore the primary
output and the class label is provisional.
