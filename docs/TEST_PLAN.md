# Acceptance Test Plan and Test Cases

Scope follows `ARCHITECTURE.md` §5 and the withdrawal of NFR-04: this plan tests **one**
capture mode, a fixed camera over a conveyor. Test cases for multi-device consistency
and fisheye fixtures have been withdrawn with the requirement they served, and are
listed at the bottom rather than deleted, so the scope change stays legible.

## Traceability

| Acceptance criterion | Requirements | Evidence |
|---|---|---|
| AC-01: valid input never crashes | FR-01, FR-02 | Fixtures and valid JSON per input |
| AC-02: mask aligns with defect | FR-02, FR-05 | IoU, clDice, visual overlay |
| AC-03: types are distinguished | FR-03 | Per-class confusion matrix, per-class recall |
| AC-04: geometry is valid | FR-04 | Hand-checked regions and `app/test_pipeline.py` |
| AC-05: clean images stay clean | FR-06, NFR-03 | `test_negatives` FP area ≤0.005 |
| AC-06: transfer is honestly measured | NFR-07 | Frozen wood test and leakage assertions |
| AC-07: deployment budget | NFR-01, NFR-02 | ONNX file size and idle-desktop latency |
| AC-08: offline/reproducible | FR-08, NFR-05, NFR-06 | Import audit and fixed-input hash |
| AC-09: batch traceability | FR-07, FR-08 | CSV/audit schema test |
| AC-10: exported model is the measured model | NFR-06 | ONNX↔torch logit drift and argmax agreement |
| AC-11: per-material claims are separable | NFR-08 | Per-material breakdown of the headline split |

## Test cases

| ID | Scenario | Expected result |
|---|---|---|
| TC-01 | Clean steel, ceramic, plastic and wood | Valid empty result; aggregate FP area ≤0.5%. |
| TC-02 | Labelled steel crack | Aligned crack mask, valid geometry, written overlay. |
| TC-03 | Held-out steel scratch | Typed `scratch`, not `crack`; confusion recorded. Scratch evidence is 624/724 steel, so this case is steel-only by construction; ceramic/plastic/glass scratch is transfer and is not asserted here. |
| TC-04 | Crack and scratch together | Correctly typed regions; no class collapse. |
| TC-05 | *(withdrawn — see below)* | |
| TC-06 | *(withdrawn — see below)* | |
| TC-07 | Belt-axis blur, specular highlight, low light, JPEG | Graceful degradation; no full-frame mask explosion. |
| TC-08 | Frozen unseen wood | Report clDice/IoU/FP without training on wood positives. |
| TC-09 | Repeated input | Same mask and byte-identical non-timestamp result. |
| TC-10 | Batch of 100 images | CSV/audit rows per image/region; no crash/memory growth. |
| TC-11 | Pseudo-label review | No pseudo row reaches evaluation; human approval recorded. |
| TC-12 | **ONNX parity** | Exported graph reproduces the torch model: max abs logit diff <1e-3 and per-pixel argmax agreement >0.999. A segmentation head is more export-sensitive than a classifier — every pixel is its own argmax, so a small logit shift moves the decision on exactly the thin structures the system exists to find. |
| TC-13 | **Per-material headline breakdown** | `test_factory` reported per material, with image-weighted and material-weighted clDice side by side. A gap between them means the headline is carried by whichever material contributes the most images, and must be disclosed. |
| TC-14 | **All-background baseline** | Reported alongside every result. A model predicting nothing scores 96–98 % pixel accuracy on `test_factory`; if the metric set cannot distinguish that from a real model, the metric set is wrong. |
| TC-15 | **Input transform is selected on `val` and audited against NFR-03** | `bench/prep_sweep.py` refuses a test split for selection. Any transform carried forward must report `fp_area` on `test_negatives`, because the whole family trades false positives for sensitivity: the best-headline candidate (`median3`, clDice +0.016) breaches the 0.005 ceiling at 0.0061 and must be rejected on that basis alone. |
| TC-16 | **Deployment applies the transform the weights were trained with** | The spec is stamped into the ONNX `metadata_props` under `prep` and read back by `app/inference.py`. Skipping it does not raise — it costs detection 0.962 → 0.949 silently, which is exactly the class of failure no metric in the benchmark would surface. |

### Withdrawn with NFR-04

| ID | Original scenario | Why withdrawn |
|---|---|---|
| TC-05 | Same part, three devices | Multi-device support was withdrawn from `REQUIREMENTS.md`. The corpus contains almost no real borescope, webcam or phone captures, so the case could only ever have been passed against synthetic distortion, which tests the augmentation rather than the model. |
| TC-06 | Fisheye fixture | The deployment lens is rectilinear and calibrated. Testing fisheye tolerance would validate a robustness the station does not need, and passing it would imply support for handheld optics that is not claimed. |

Both return if handheld capture is ever revisited, together with the device-validation
study `ARCHITECTURE.md` §5 requires. The `handheld` camera profile still exists in
`bench/camera_aug.py` for exactly that reason.

## Gates and reporting

```bash
# dataset must regenerate from source and pass strict QA before any training claim
./.venv/Scripts/python.exe dataset/split.py
./.venv/Scripts/python.exe dataset/qa.py --strict

# three seeds minimum, then aggregate on the tag
bash run_v9.sh
./.venv/Scripts/python.exe bench/summarize.py --tag V9

# per-material breakdown of the headline split (AC-11 / TC-13)
./.venv/Scripts/python.exe bench/per_material.py --tag V9_22

# export: --classes 3 and --tag are mandatory, or a three-class export would
# overwrite the binary one and every stored size/latency figure would describe a
# model that no longer sits at that path. Run only on an idle machine.
./.venv/Scripts/python.exe bench/export_onnx.py \
  --models smpslim_timm-mobilenetv3_small_100 --classes 3 --tag v9 \
  --weights data/bench/smpslim_timm-mobilenetv3_small_100_V9_22.pt
```

Report mean ± standard deviation over at least three seeds, split version and SHA with
each metric, and no mixed crop/whole-image protocol. A failed QA assertion blocks
training claims.

Three reporting rules exist because each was broken at least once during development:

1. **Single-seed numbers are not evidence.** Cross-material clDice was measured at
   0.579 / 0.309 / 0.075 across three seeds of an otherwise identical configuration.
2. **Quote the sample size with any split under ~50 images.** `test_factory_scratch` is
   31 images and `test_factory`'s glass subset is 2; a clDice on the latter is noise
   with a decimal point.
3. **Class-agnostic geometry must never stand in for class accuracy.** Headline clDice
   and IoU are computed on `any_defect`. A model can score 0.67 clDice while typing
   more than half of crack pixels as scratch — which is the currently measured state.

Latency is a desktop measurement only. No ARM device has been measured, so no phone or
edge latency is claimed; the corresponding target was withdrawn from NFR-02 rather than
carried as an untested number.
