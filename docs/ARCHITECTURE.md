# Surface-Defect Inspection — Architecture

Expands Phase 1 §12 with what was built and measured. Requirement IDs are Phase 1's
(`FR-nn`, `NFR-nn`), test IDs are Phase 2's (`T-nn`); `REQUIREMENTS.md` holds their
status and `DECISIONS.md` the reasoning behind each choice. Where this document and
Phase 1 disagree, this one is what the code does, and the difference is recorded in
`REQUIREMENTS.md` §4.

Section order: §1–5 the problem and the input path, §6–7 the model and post-processing,
§8 deployment and interfaces, §9 human judgement, §10 status, §11 modularity and
scalability.

## 1. Overview

This system performs automated visual inspection of finished products on a factory conveyor line, using a fixed industrial camera mounted above the belt. It segments the surface into crack and scratch regions at pixel level, measures the geometry of each region, and produces a structured record per item.

The system automates the search step of inspection—scanning the full surface of every product for possible defects—which is the part of manual inspection that causes the most eye fatigue when done continuously by a human operator. It does not automate the judgement step: a flagged region and its geometry are handed to a human for confirmation, because validating an already-located candidate region is a much smaller visual and cognitive load than searching an entire surface unaided. Automatic validation of flagged regions is a capability the architecture can support, but it is not the default behaviour; validation remains a human step until flagged-region accuracy is characterised well enough to automate it.

Inference runs on local edge compute at the line with no network dependency. Results are viewed separately through a laptop or phone interface, described in Section 8.

## 2. System Pipeline

```text
fixed industrial camera (conveyor line)
        |
   Acquisition — decode frame, attach product/line metadata
        |
   Input preparation — orientation -> resize to 256×256 -> input transform -> normalise
        |
   Segmenter — MobileNetV3-Small encoder + lightweight U-Net decoder
        |
   3-class output: background | crack | scratch
        |
   Post-processing — thresholding -> connected components -> region geometry
        |
   Result assembly — one structured record per product
        |
   Outputs: overlay image | batch CSV report | audit log entry | laptop/phone report view
```

Batch processing runs this same pipeline once per stored image; it is not a separate implementation. The conveyor-line path and batch path use identical normalisation, model weights, thresholds and post-processing configuration, so an offline batch report is directly comparable with a line result.

## 3. Material Considerations

None of the components in this document—the segmentation network, post-processing pipeline or model configuration—depend on a specific target material. Moving to a new material is a matter of collecting material-specific images, freezing a representative test split and retraining or fine-tuning the weights; it does not require restructuring the pipeline.

The table below describes the deployment trade-off when choosing which material to support on a specific factory line. “Public data” means data useful to the current task, not merely images that contain a defect.

| Material | Public data useful to pixel segmentation | Product type | Deployment implication |
|---|---|---|---|
| Steel | SteelDefectX supplies thin crack/scratch masks; Severstal, NEU-DET and GC10-DET mainly supply clean/hard-negative steel/metal texture | Finished sheet, coil, fabricated or machined part | First priority: strongest factory evidence and largest Indian manufacturing relevance |
| Ceramic | Limited crack masks; magnetic-tile data and ceramic subsets are available | Tile, sanitary ware, tableware | Useful second material, but real production-line captures remain necessary |
| Glass | 12 real masks in the whole corpus | Finished glass, screen, mirror | **Not supported.** 12 masks cannot support a per-material metric; requires customer-collected images before any claim |
| Plastic | 936 real masks, all PVC pipe from one source | Moulded part, casing, packaging component | Gap closed for *one product*, not for the material. A moulded casing line is not covered by pipe data |
| Epoxy | Very small paired-mask source from commutator embedding | Electrical component / resin finish | Treat as a narrow material, not as a proxy for all plastics |
| Concrete | Extensive civil crack data | Installed/structural material, not a conveyor-line product | Auxiliary morphology data only, not factory deployment evidence |

The first deployment should be material-specific: a steel line uses steel training and a held-out steel test set; a plastic line is not considered supported merely because the same architecture exists. This prevents a material-agnostic architecture from becoming a material-agnostic claim.

## 4. Defect Types

| Defect type | Typical shape | Typical width | Notes for detection |
|---|---|---|---|
| Crack | Thin, often curved or branching linear structure | Narrow, often a few pixels at the model resolution | May be low contrast depending on surface finish; benefits most from full-resolution skip connections |
| Scratch | Linear surface mark, generally straighter and more uniform than a crack | Often wider than a crack on average | May be bright or dark and can be confused with reflections, seams or brushed texture |

Both are line defects rather than area defects. Broader area/texture anomalies such as pinholing, crazing, discoloration, corrosion and dents are out of scope for this model. Including them would require the model to learn a materially different visual signature, dilute annotation effort and make crack/scratch performance harder to interpret.

## 5. Image Acquisition

Image acquisition is assumed to use a fixed industrial camera that captures each product at a constant working distance and controlled illumination as it passes through the inspection zone. The acquisition service emits one selected frame per product with timestamp, line/station identifier and optional product/batch identifier.

Because the line setup is fixed rather than handheld, no per-device capture branching is required downstream. Camera configuration is recorded once in the station profile: resolution, exposure, gain, lens, working distance, light arrangement and any geometric calibration. The inference path preserves the original frame and its metadata, then produces a 256×256 model view as described in §5.1.

The inspection station must be commissioned with three practical checks:

1. **Field-of-view check:** the complete inspectable product surface is present in the selected frame, with no frequent belt/fixture occlusion.
2. **Focus and exposure check:** the smallest defect of interest remains visible under the expected product finish and conveyor speed.
3. **Scale check:** if millimetre reporting is required, a physical reference/calibration target is captured at the inspection plane. Without it, geometry is reported only in pixels.

The fixed-camera assumption is a deployment choice, not a limitation of the core model. Phone, webcam and borescope images can use the same model interface later, but they require a separate device-validation study before being treated as supported deployment modes.

**This is now the binding scope, not a preference.** `REQUIREMENTS.md` NFR-04 originally required one model covering borescope, webcam and phone; it has been formally withdrawn, because the corpus contains almost no real captures from those devices and the requirement could only have been asserted, never demonstrated. One consequence is visible in `bench/camera_aug.py`: the `handheld` profile still exists and still models perspective, barrel distortion, rolling-shutter shear and chromatic aberration, but it is no longer the default. It is retained so pre-v5 benchmarks stay reproducible and so handheld capture can be revisited without rewriting the augmentation — not because those modes are supported.

### 5.1 Input preparation

Four steps turn a captured frame into a model tensor. They are fixed, deterministic, and
identical in training and deployment — preprocessing that differs between the two is an
accuracy loss no metric in the benchmark can reveal, because both sides of every
measurement would shift together.

| # | Step | Detail |
|---|---|---|
| 1 | Resize | `cv2.resize(frame, (256,256), INTER_LINEAR)` — **plain resize, not letterbox.** Matching the training loader matters more than preserving aspect ratio |
| 2 | **Input transform** | `cv2.bilateralFilter(img, 5, 50, 50)` on the BGR uint8 frame |
| 3 | Scale | BGR → RGB, `/255` |
| 4 | Normalise | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, then `HWC → CHW` |

Step 2 is new in v10 and is the subject of the rest of this section.

**Placement.** In training the transform runs *after* camera augmentation, not before.
The deployed system filters a frame the camera has already degraded — belt-axis motion
blur, specular highlights, sensor noise — so the transform must see that degradation
during training too. Applying it to a clean patch and then adding noise would train a
filter-then-corrupt pipeline that never occurs on the line.

#### Why a transform at all, and how this one was chosen

`bench/prep_sweep.py` swept 22 configurations — 13 single transforms and 9 deliberately
ordered pairs — against an unmodified baseline on identical crops. Selection is on
`val`; the script refuses a test split outright, for the same reason §7.1's thresholds
are chosen on validation data. Winners are then applied to the frozen splits.

The central finding is that **the whole denoising family is a single sensitivity knob.**
Detection, crack class recall, unseen-material response and clean-surface false-positive
area all move together, in the same direction, in proportion to how aggressively the
transform smooths. There is no setting that improves detection without also spending
false positives. The engineering question is therefore not "which transform is best" but
"which point on that curve stays inside NFR-03".

| Transform | `test_factory` clDice | detect | `test_negatives` fp_area | Verdict |
|---|---:|---:|---:|---|
| none | 0.7197 | 0.949 | 0.00286 | baseline |
| **bilateral (d=5, σ=50/50)** | **0.7262** | **0.962** | **0.00396** | **selected** |
| bilateral (d=7, σ=75/75) | 0.7142 | 0.962 | 0.00425 | inside NFR-03, worse headline |
| median 3×3 | 0.7357 | 0.968 | 0.00610 | **rejected — breaches NFR-03** |
| median 5×5 | 0.7121 | 0.968 | 0.00643 | rejected |
| Gaussian σ=1 | 0.7253 | 0.968 | 0.00873 | rejected |

`median3` produced the best headline of anything measured — clDice +0.016, crack class
recall 0.429 → 0.460 on the one metric the project has struggled to move — and was
rejected anyway. It pushes clean-surface false-positive area to 0.61 %, through the
0.5 % ceiling that v9 had only just met for the first time. A requirement that is
abandoned the moment it becomes expensive was never a requirement.

`bilateral` takes the smaller headline gain (+0.007) and stays at 0.40 %. Its stronger
justification is elsewhere: unseen-material clDice +0.088 with detection 0.436 → 0.584,
which recovered a substantial part of the transfer loss documented in §10.3. That result
also identified the mechanism — a meaningful share of that loss was noise sensitivity,
not a lost representation.

#### Two results worth recording

**CLAHE, the reflex choice for surface inspection, is among the worst options measured**:
−0.057 clDice at clip 2.0, −0.108 at clip 4.0. It amplifies local contrast everywhere,
including on defect-free surface texture, manufacturing crack-like evidence that the
model then believes. The standard image-processing recipe for making a defect visible to
a *human* is close to the opposite of what helps a learned thin-structure segmenter.

**No complementary pair exists.** Every one of the nine pairs landed at or below its
weaker component; `clahe2+blackhat` reached −0.21 against −0.057 and −0.051 alone, and
the best pair, `median3+unsharp_soft`, scored −0.002 — the sharpening cancelling the
denoising exactly. The reason is structural: these operators all manipulate local
contrast, so composing them compounds one distortion rather than correcting two
independent nuisances. Pairs were ordered deliberately (denoise before sharpen, flatten
before contrast) and reversing the order only makes it worse.

Two of the four transforms that won on `val` lost on the test splits — `gamma 0.7` by
0.015 and illumination flattening by 0.084. On margins this small, `val` ranks the
family correctly but not the individuals, which is precisely why selection is not
allowed to touch a test split.

#### Cost and binding

The filter adds **0.99 ms** (p90 1.27 ms) per 256×256 frame on one desktop thread,
against 26 ms for inference — under 4 % of the budget.

The transform is part of the trained configuration, not a deployment option.
`bench/export_onnx.py` stamps its specification into the ONNX graph's `metadata_props`
under the key `prep`, and `app/inference.py` reads it back and applies it. A checkpoint
trained with a different transform therefore changes the deployment path automatically,
and an export carrying no `prep` key means raw input — which is how every model before
v10 was trained, so earlier artefacts stay valid. Omitting the filter does not raise an
error; it silently costs detection (0.962 → 0.949) and about half the model's response
on materials outside the training set.

## 6. Model Architecture

| Component | Role |
|---|---|
| MobileNetV3-Small encoder (ImageNet-pretrained) | Extracts features at multiple resolutions; chosen for low parameter count and CPU latency relative to larger backbones |
| Lightweight U-Net decoder (channel widths 64, 48, 32, 24, 16) | Reconstructs a full-resolution segmentation mask from encoder features |
| Full-resolution skip connections | Carry early, high-resolution encoder features into the decoder so structures only a few pixels wide are not lost at deeper strides |
| Output head | Produces a 3-class per-pixel score: background, crack, scratch |

The model takes a normalised tensor of shape `3 × 256 × 256` and returns `3 × 256 × 256` logits. A softmax converts these logits into mutually exclusive per-pixel class scores. A pixel can therefore be background, crack or scratch, but cannot be counted as both defect types.

| Configuration | Parameters | Model size (ONNX) | CPU latency, one desktop thread | Architectural decision |
|---|---:|---:|---:|---|
| Lightweight custom network, no pretrained backbone | 0.14 M | 0.6 MB | 13.3 ms | Rejected: small and fast, but produced 10–15× higher false-positive area on clean surfaces |
| **MobileNetV3-Small + lightweight U-Net** | **1.43 M** | **5.8 MB** | **26.1 ms** | Selected binary baseline and starting topology for the three-class model |
| MobileNetV3-Large + lightweight U-Net | 3.74 M | 15.1 MB | 34.5 ms | Larger than the intended deployment budget |
| MobileNetV3-Small + full U-Net decoder | 3.59 M | 14.4 MB | 55.4 ms | Decoder cost is too high for the intended edge profile |
| EfficientNet-B0 baseline | 6.25 M | 23.4 MB | 71.0 ms | Accuracy/architecture reference, not line deployment candidate |

The custom network without a pretrained backbone was not carried forward despite its smaller size because clean-surface false positives are operationally more harmful than a modest model-size increase. MobileNetV3-Small with the lightweight decoder gives the best measured balance of size, latency and thin-structure performance for the current edge budget.

**Status: the three-class head is implemented and measured.** It reuses the selected encoder/decoder topology unchanged and alters only the final head, loader, loss and metrics; the binary path is retained behind `--classes 1` so every stored benchmark above remains reproducible. The size and latency figures in the table are still the binary export's — a three-class head changes the final 1×1 convolution only, so parameter count moves by ~0.03 %, but the numbers will be re-measured on the exported three-class model rather than assumed.

Binary benchmark numbers must still not be presented as three-class crack/scratch results. The two heads answer different questions, and the three-class head measures one thing the binary head structurally could not: **class confusion**. The binary model fired 3.7× harder on real scratches than on real cracks, and with a single foreground class that damage was invisible — it registered as a good detection rate.

Training for the three-class model uses class-weighted cross-entropy plus soft Dice loss. Cross-entropy makes the classes compete per pixel; Dice supplies useful gradient when a defect covers very little of an image. Class weights are computed from the observed training distribution because cracks and scratches differ in frequency and foreground fraction. Training uses AdamW, fixed seeds, mixed precision where available and exponential-moving-average weights; validation and final testing use frozen, real images without random augmentation.

The scratch class is rebalanced at the **sampler**, not in the loss. At its natural 6 % frequency it was never predicted at all, and raising its loss weight instead buys recall by over-painting, which is the failure NFR-03 exists to catch.

Long runs use a cosine learning-rate schedule decaying to 2 % of the initial rate. At a constant rate the headline metric swung ~0.06 between adjacent epochs while validation stayed flat — the optimiser orbiting a minimum rather than converging. Over a long run that also corrupts model selection: choosing the best of 80 epochs scored on a noisy series is selection on the evaluation split, and the reported figure would be biased upward by whichever epoch was luckiest.

The input transform described in §5.1 is part of this configuration, not a wrapper around it. The model is trained and evaluated on transformed input alike.

## 7. Post-Processing and Region Geometry

The per-pixel output is converted into a region-level result through a deterministic, versioned profile:

1. **Class decision:** softmax scores are compared with class-specific foreground thresholds selected on validation data. A pixel below both foreground thresholds is background; if both pass, the higher score wins.
2. **Connected components:** components are calculated separately for crack and scratch using 8-connectivity, so adjacent defects of different types remain distinct.
3. **Noise filtering:** regions below a configured minimum pixel area and/or minimum skeleton length are discarded before measurement. These limits are chosen to meet the clean-surface false-positive-area target, not by tuning on final test images.
4. **Area:** pixel count of the final region. Physical area is reported only if the station has valid scale calibration.
5. **Length:** medial-axis skeletonisation followed by 8-connected arc-length summation. This measures curved and branching defects more faithfully than a bounding-box diagonal.
6. **Maximum width:** Euclidean distance transform inside the component is sampled along its skeleton; twice the maximum distance is the widest local point. This is more informative than area divided by length, which hides a locally wide section.
7. **Overlay:** region masks, IDs and type labels are transformed back to original image coordinates and rendered as a semi-transparent overlay for review.

All thresholds, connectivity and filtering limits are included in the processing configuration recorded with the result. Re-running the same input with the same model and configuration produces the same regions and geometry.

## 8. Deployment and Interfaces

| Component | Role | Status |
|---|---|---|
| Edge inference unit, line-side | Runs the selected ONNX model, post-processing and record assembly within the line throughput budget | Three-class checkpoint trained; post-processing and batch record assembly implemented; ONNX export pending an idle machine |
| Int8 deployment model | Reduces model size and potentially improves edge latency | Planned; accepted only after regression against float32 |
| Float32 reference model | Reference for quantisation sensitivity, especially near the minimum detectable defect size | Binary float32 export exists; three-class export is the next artefact, and it self-verifies against the torch model on logit drift and per-pixel argmax agreement before it may be shipped |
| Laptop/phone interface | Displays overlay images, batch CSV reports and audit records created by the edge unit; not used for capture | Presentation layer only. The Streamlit prototype is a placeholder and is being replaced; the batch CLI and the shared post-processing module are the supported interface |

Every inference result records the model version, image hash, station ID and processing configuration, so it can be traced to the exact model and settings that produced it. The edge unit has no network requirement: the laptop/phone interface reads local output records or a controlled local connection; it never recomputes segmentation itself.

### 8.1 Result schema

```json
{
  "schema_version": "1.0",
  "image_sha256": "…",
  "model_version": "…",
  "station_id": "…",
  "product_id": "optional-local-id",
  "material": "optional-declared-material",
  "processed_at": "ISO-8601 timestamp",
  "processing_profile": "versioned-profile-id",
  "regions": [
    {
      "id": 1,
      "type": "crack | scratch",
      "area_px": 0,
      "length_px": 0.0,
      "max_width_px": 0.0,
      "bbox_xywh": [0, 0, 0, 0]
    }
  ],
  "empty": true,
  "overlay_path": "optional-local-output-path"
}
```

A defect-free product is recorded explicitly with `"empty": true` and an empty region list, so a clean inspection and a failed/missing capture are never ambiguous. An unreadable image produces a structured acquisition failure instead of an empty successful record.

### 8.2 Batch Processing and Audit Logging

Given a directory of stored images, the pipeline runs once per file and concatenates results into a CSV report: one row per detected region and one explicit row for each defect-free image. Every processed image also appends a local audit record containing image hash, model version, station ID, configuration profile, timestamp and success/failure state. This produces a traceable inspection history without a cloud service.

## 9. Human Judgement and Validation

Model scores are used internally for thresholding and ranking candidate regions, but are not displayed as probability percentages. What reaches the operator is a flagged region with geometry and overlay, not an uncalibrated “95% confidence” value.

The reason for keeping judgement primarily human is where the cognitive load falls. Unaided search across every product surface, continuously, produces fatigue and inconsistent attention. Confirming or dismissing a small number of already-located candidate regions is substantially lighter than finding those regions in the first place. The system takes on the search load and leaves the lighter validation load to the operator by default.

Automatic validation—the system deciding that a flagged region is genuinely defective without confirmation—is a separately validated capability, not default behaviour. It requires a held-out flagged-region study, calibrated confidence, false-alarm analysis by material/line, and agreed customer tolerances. The architecture retains scores and geometry needed for that extension without pretending that those conditions have already been met.

## 10. Implementation and Validation Plan

The architecture is delivered in staged increments so each step remains testable against the factory use case.

| Stage | Work | Deliverable | Acceptance evidence |
|---|---|---|---|
| 1. Three-class model | Replace binary head with background/crack/scratch head; add class-aware loader, loss and metrics | Trained three-class checkpoint and per-class report | Scratch/crack confusion matrix, clDice/IoU per class, clean-surface FP area |
| 2. Inspection service | Build image-to-record path: preprocessing, ONNX inference, regions, geometry, overlay, JSON and CSV | Local CLI/service processing a directory | Every input produces a record or explicit failure; geometry/overlay spot checks |
| 3. Edge packaging | Export float32 ONNX, then int8; implement model/version/profile packaging | Reproducible deployment artefacts | Size, latency and float32-vs-int8 accuracy/FP regression report |
| 4. Station commissioning | Capture line-specific clean and defect fixtures; set field of view, exposure and scale | Versioned station configuration | Smallest target defect visible; clean-line FP test; pixel-to-mm validation if needed |
| 5. Human review loop | Present flagged crops/overlays, collect confirmations and failure cases | Review workflow and error log | Operator review time, false alarms and missed-defect cases recorded |
| 6. Optional automation | Consider automatic validation only after Stage 5 evidence is sufficient | Separate validated decision module | Calibrated score study and customer-approved decision thresholds |

### 10.1 Model-development plan

The three-class model is trained on real crack and scratch masks plus controlled synthetic training data. Synthetic examples and pseudo-labelled images remain training-only; validation and final test use frozen real images. Model selection reports thin-structure metrics (clDice and tolerant F1), overlap (IoU), crack/scratch confusion and false-positive area on clean surfaces. Pixel accuracy is not used as a headline metric because an all-background model can appear accurate while finding no defects.

The first final deployment target is steel. It has the strongest current factory evidence and the largest immediate line relevance. Plastic, non-steel metal, glass and pipe surfaces are not declared supported until each receives its own representative real-image training/validation/test collection.

### 10.2 Quantisation and latency plan

The float32 reference model must be measured on an idle desktop CPU and the actual intended edge device. Int8 quantisation is then evaluated using the same frozen image suite. It is accepted only if it stays within the size/throughput budget and does not materially worsen per-class segmentation or clean-surface false-positive area. Desktop latency is a development measurement only; it is not substituted for line-side or ARM evidence.

### 10.3 Current implementation status

| Architecture element | Status |
|---|---|
| Dataset adaptation, frozen splits, leakage QA and camera augmentation | Implemented. Splits v7, 80:15:5 stratified by material × foreground quartile, sha `c0fde17c96749567`, `dataset/qa.py --strict` passing |
| Binary MobileNetV3-Small + slim U-Net benchmark and float32 ONNX export | Implemented and measured |
| Three-class head, class-aware loss/loader and per-class training | **Implemented and measured.** Best single run to date: `test_factory` clDice 0.720, IoU 0.518, detection 94.9 %, clean-surface FP area 0.33 % |
| Input transform selection (§5.1) | **Implemented and measured.** 22 configurations swept on `val` and reported on the frozen splits; `bilateral` selected on the NFR-03 constraint, stamped into the ONNX metadata and applied by the app |
| Transfer to materials outside the training set | **Open, and characterised.** Extending training from 4 to 20 epochs improved every in-distribution metric and halved response on unseen material (wood clDice 0.73 → 0.27). A threshold sweep from 0.50 down to 0.10 moves detection there by 0.005, so the loss is representational, not a decision boundary, and cannot be recovered at inference. `bilateral` recovers part of it (+0.088 clDice) |
| Region geometry, overlay, JSON/CSV batch tool and audit log | Implemented in `app/postprocess.py` and `app/batch.py`, with a test asserting the interactive and batch paths produce identical geometry. The interactive UI is a placeholder pending replacement |
| Three-class ONNX export with torch-parity verification | Implemented in `bench/export_onnx.py`; export blocked until the machine is idle, because latency measured under contention is meaningless |
| Int8 export and float32-vs-int8 regression test | Planned |
| Station-specific commissioning and edge/ARM latency measurement | Planned. **No ARM measurement exists**, and the corresponding phone latency target has been withdrawn from `REQUIREMENTS.md` rather than left as an unevidenced claim |
| Calibrated, user-facing confidence and automatic validation | Deferred pending a separate validation study |

**Known open defect.** Crack/scratch class recall on `test_factory` sits at 0.43–0.49: the model detects defects reliably but assigns the wrong *type* to more than half of crack pixels, biased toward scratch. Because the headline clDice and IoU are class-agnostic, this does not appear in them — it is visible only in the confusion metrics this head exists to produce. FR-05 is therefore **partially met**, and the geometry results should not be read as validating the type label.

This distinction keeps the architecture ambitious but honest: the implementation plan is explicit, and planned capabilities are not presented as completed functionality.

## 11. Modularity and scalability

### 11.1 Layering

Four layers, each depending only on the one below. The dependency direction is the design:
nothing in the pipeline imports the interface, so the interface can be replaced — as it is
being replaced — without touching a measured path.

```text
interface        Phase 2 §4 screens                     (separate deliverable)
application      batch runner, services, record writer  app/batch.py
inference core   preprocess -> ONNX -> class map -> geometry -> record
                                                        app/inference.py, app/postprocess.py
evidence base    dataset construction, training, evaluation, export
                                                        dataset/, bench/, defectforge/
```

`bench/` may import from `app/` — `class_thresh.py` and `final_eval.py` deliberately
evaluate through `app.postprocess.class_map` — but `app/` never imports from `bench/`
except for the one input transform read out of the model metadata. **A sweep that chose
thresholds under a private copy of the decision logic would be tuning a model that never
runs.**

### 11.2 Single sources of truth

Each of these was a duplicate at some point, and each duplicate drifted before it was
removed. They are listed because the failure mode is identical every time: two copies of
one rule, both plausible, no test able to see the difference.

| Rule | Single implementation | What drifted before |
|---|---|---|
| §7.1 class decision | `app/postprocess.py::class_map` | Evaluation used argmax while the app used thresholds, so reported accuracy described a model nobody ran |
| Post-processing thresholds | `app/postprocess.py::Profile` | The batch CLI hard-coded 0.50/0.50 and silently overrode the values chosen on `val`. The override flags are now removed |
| Input pipeline | `app/inference.py::preprocess`, mirroring `bench/data.py` | Nothing yet — and T-02, the case that would catch it, is the most valuable missing test |
| Input transform spec | ONNX `metadata_props["prep"]` | Nothing yet; stamped into the artefact so a sidecar cannot be separated from the model |
| Run → claim mapping | `docs/RESULTS.md` | Numbers quoted from runs whose command had been thrown away |

### 11.3 What has to change when something changes

| Change | What has to change |
|---|---|
| New weights after retraining | Swap the `.onnx` and record the hash. No code change unless input/output shape changes |
| Re-tune thresholds | `Profile` in `app/postprocess.py`. Nothing else — the interface reads it at start-up (T-06) |
| Different input transform | `--prep` at export. The app reads it from the graph (T-29) |
| Swap ONNX Runtime for another CPU runtime | The session wrapper only. The model is a portable graph, not a runtime-specific format |
| Add a third defect class | Retrain with a wider head, one row in `defect_class`. The post-processing loop is already per-class, and the metrics are already per-class |
| Add a new material | A material profile and its own real test slice. No pipeline restructuring — nothing in the model or geometry depends on the material |
| Add a verdict later | One module between geometry and the record, plus a column on `inspection`. Nothing downstream assumes a verdict exists (ADR-020) |
| Add a new data source | An adapter in `dataset/adapters.py` plus a row in `sources.yaml`; re-run the four-command gate in `DATASET.md` §9 |

### 11.4 Scale

| Dimension | Position |
|---|---|
| Throughput | 26 ms inference on one CPU thread leaves headroom for roughly one part per second per station once the rest of the pipeline is measured. Beyond that, more worker processes on the same machine |
| More stations | One unit per station, sharing nothing at runtime, so stations do not contend |
| Storage | ~1 KB per inspection including regions; roughly 30 MB/day at one part per second. Overlays kept 48 h when clean, 90 days when regions were found |
| Dataset | The manifest is a single CSV at 51,504 rows and the loader indexes it by integer position rather than copying frames per worker — the copy is what exhausted the Windows commit limit at 4 workers. Growth is linear; the phash leakage audit is the quadratic step and is the first thing that needs attention past ~100 k images |
| Model | Measured, not assumed: no reliable accuracy gain above 1.43 M parameters across nine architectures (ADR-003). Scaling capacity is not the lever; per-material data is |
| Evaluation | Splits are frozen with a sha and regenerate from source, so adding data does not invalidate past results — it creates a new freeze that is compared against the old one explicitly (`RESULTS.md`) |

### 11.5 Sustainability constraints worth stating

- **Non-commercial data.** KolektorSDD2 and MVTec AD are CC BY-NC-SA 4.0, and Severstal comes under Kaggle competition terms. Fine for a prototype, blocking for a product: a commercial version must retrain on licensed or self-collected images. A model trained on NC data is not automatically free of the NC condition.
- **No CI yet.** The suite in `TEST_PLAN.md` §4 runs by hand. That is the largest sustainability gap in the repository.
- **Two-person team.** Scope was deliberately kept to one active workstream per person; the interface and the store are the only new components in Phase 2 because the pipeline already existed.
