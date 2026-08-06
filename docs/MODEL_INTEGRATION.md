# Model Integration

How the interface talks to the inspection pipeline, and what to do when the final ONNX
export lands.

---

## The one rule

**Everything goes through `app/inference.py::Inspector`.**

The accuracy figures this project reports are the numbers that exact path produces. A
second implementation that resizes differently, or applies argmax instead of the
per-class thresholds, is a different system with different accuracy — and no benchmark
in the suite would catch the difference, because every benchmark runs through
`Inspector` too.

So the application layer never:

- imports or calls `onnxruntime`
- resizes, scales or normalises an image
- applies softmax
- compares a probability against a threshold
- labels connected components or measures geometry
- downscales the overlay

`app/providers/real_provider.py` is the only module that knows the pipeline exists, and
all it does is translate key names and write files. It performs no image mathematics.

---

## The boundary

```python
class InspectionProvider(Protocol):
    name: str
    def inspect(self, image_bgr, image_bytes, product_id: str, material: str) -> InspectionResult: ...
    def describe(self) -> dict: ...
```

Two implementations:

| Provider | Module | When |
|---|---|---|
| `MockInspectionProvider` | `app/providers/mock_provider.py` | `INSPECTION_PROVIDER=mock` (default) |
| `RealInspectionProvider` | `app/providers/real_provider.py` | `INSPECTION_PROVIDER=real` |

Selection happens in `app/providers/__init__.py::build_provider`. In mock mode the real
provider module is **never imported**, so `onnxruntime` is not a mock-mode dependency —
which is why `requirements.txt` leaves it commented out.

`InspectionResult` is what both return: status, regions, image paths, provenance, and
the canonical record. Services, routes and templates are written against that type and
cannot tell which provider produced it.

---

## Switching to the real model

```env
INSPECTION_PROVIDER=real
MODEL_PATH=data/export/smpslim_timm-mobilenetv3_small_100_v8a.onnx
MODEL_SHA256=9f2c...ea41
STATION_ID=line-1-cam-A
```

Install the inference dependencies (uncomment them in `requirements.txt`):

```bash
pip install onnxruntime opencv-python-headless scikit-image
```

At start-up the real provider:

1. checks the file exists — otherwise `ProviderUnavailable(code="model_file_missing")`
2. computes its SHA-256 and compares it with `MODEL_SHA256` — a mismatch raises
   `model_hash_mismatch` and **inspection does not start** (NFR-13 / T-23)
3. imports `Inspector` lazily — a missing inference core raises `inference_core_missing`
4. constructs `Inspector(model_path, station_id=...)`

Every failure is surfaced on the Status screen with the reason and a recommended action,
and the health strip on every other screen stops presenting the station as healthy.

---

## The ONNX contract

The application is written against this contract, not against a particular set of
weights. Retraining swaps the file and records the new hash; no code changes unless the
input or output shape changes.

| Property | Value |
|---|---|
| Input name / shape | `input`, `(1, 3, 256, 256)` float32, batch axis dynamic |
| Channel order | **RGB**, not BGR |
| Preprocessing | plain resize to 256×256 `INTER_LINEAR` → `/255` → ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` → CHW |
| Output name / shape | `logits`, `(1, 3, 256, 256)` float32 |
| Channel meaning | `0 = background`, `1 = crack`, `2 = scratch` |
| Activation | none in-graph — **raw logits**, softmax applied by the caller |

Two things bite people here, and both are inside `Inspector`, not here:

- The model is trained on **plain resize, not letterbox**. Matching the training loader
  matters more than preserving aspect ratio; preprocessing that differs between training
  and deployment is a silent accuracy loss.
- The graph emits **logits, not probabilities**. Feeding raw logits into a threshold
  meant for probabilities appears to work while being badly miscalibrated.

Note the channel meaning is the *class* taxonomy. Section 3.4 of the Phase 2 report has
a typo listing it as "0 = red, 1 = green, 2 = blue"; `INTEGRATION.md` and the rest of
that report are consistent that it is background/crack/scratch.

## The class decision is not argmax

Each foreground class is compared against its own floor. A pixel below both is
background; where both pass, the higher score wins.

```python
prob = softmax(logits)
crack_pass   = prob[1] > Profile.crack_thresh
scratch_pass = prob[2] > Profile.scratch_thresh
```

Argmax is measurably worse — crack recall 0.460 against 0.491 at identical clDice — and
has no background floor at all: a pixel scoring 0.34 crack / 0.33 scratch / 0.33
background comes out as a confident crack.

**This rule lives in the pipeline. Do not re-implement it in a service or a template.**

## Thresholds have exactly one home

`app/postprocess.py::Profile`:

```python
crack_thresh    = 0.40
scratch_thresh  = 0.20
min_area_px     = 24
min_skeleton_px = 6
```

Selected on the *validation* split, never on a test split. The region floors come from
the clean-surface false-positive budget (NFR-03).

The interface reads these at start-up and stores them on every `profile` row, so an old
inspection still resolves to the thresholds that produced it. Re-tune the module and
re-seed the profile, and the database, the API and every screen follow with no other
edit — `tests/test_materials.py::test_retuning_the_module_flows_through_to_the_interface`
proves it, and `tests/test_content_rules.py` fails the build if any threshold literal
appears in a template or a script.

---

## Record mapping

`Inspector.inspect()` returns `(record, overlay_bgr, class_map)`. The adapter maps the
record onto the application schema and preserves the original verbatim in
`result.record`, which is what gets appended to the record log.

The names in **bold** are what `app/postprocess.py::extract_regions` actually emits; the
rest are tolerated spellings kept for other record producers.

| Application field | Source |
|---|---|
| `status` | not emitted by `build_record` — derived from whether regions exist |
| `regions[].region_index` | **`id`** / `region_index` / `index` |
| `regions[].class_code` | **`type`** / `class_code` / `class` / `class_name` / `label`, or a channel index |
| `regions[].area_px` | **`area_px`** / `area` |
| `regions[].length_px` | **`length_px`** / `length` |
| `regions[].max_width_px` | **`max_width_px`** / `max_width` / `width_px` |
| `regions[].bbox` | **`bbox_xywh`** / `bbox` / `bounding_box` / `box`, as a dict or a 4-sequence |
| `regions[].centroid` | `centroid` / `center` / `centre` — **not emitted**, so it stays NULL |
| `latency_ms` | **not emitted** — the adapter times the call itself |
| `image_sha256`, `product_id`, `material`, `station_id`, `processed_at`, `empty` | same key |

> ### What the merge found
>
> This table used to carry a warning that the aliases were a best-effort guess, because
> the pipeline was not available when the interface was written, and that verifying them
> was "the single highest-risk item in the handover". It was right to.
>
> **Three of them were wrong.** The pipeline emits `type`, `bbox_xywh` and `id`; the
> adapter looked for `class_code`, `bbox` and `region_index`. None of that raises —
> `_pick` falls through to its default — so every region would have been labelled
> **crack** at bounding box **(0, 0, 0, 0)**, and a scratch would have been silently
> retyped. The result looks entirely plausible on screen: a region list with sensible
> areas and lengths, and a class label that is simply wrong.
>
> Two more gaps were quieter. `build_record` emits no `status`, which the fallback
> already handled correctly, and no `latency_ms`, so every real inspection would have
> recorded a blank latency and the health strip would have shown nothing.
>
> `tests/test_real_provider_mapping.py` now pins every key name and runs regions from
> the real `extract_regions` through the real adapter, so the next weight export cannot
> reintroduce the same class of failure.

## Overlay and class map

The overlay is written at **whatever resolution `Inspector` produced** — BGR to RGB
channel order only, because PNG writers expect RGB. No resize, no re-draw. The overlay
must register to the source image, not to the 256×256 model input (T-13).

The class map is accepted and not persisted: the region geometry derived from it is what
the database stores. If class maps need archiving later, that is a new column and a new
writer, not a change to this boundary.

---

## Swapping in new weights

The integration itself is done. Retraining produces a new `.onnx`, and that is all that
changes:

```bash
# 1. drop the export into data/export/, then record it
.venv-app/Scripts/python.exe -m scripts.register_model \
    --model data/export/<file>.onnx --version v13 --params 1430000 --latency-ms 26

# 2. point the app at it
INSPECTION_PROVIDER=real MODEL_PATH=data/export/<file>.onnx \
    .venv-app/Scripts/python.exe -m app.main
```

`register_model` writes the hash, size and version into `model_version` and reads the
contract back out of the graph — input size, class count and input transform — via
`Inspector.describe()`, so the registration cannot disagree with how the pipeline will
actually feed the model. Parameter count and latency are **not** inferred: latency is a
measurement, and one taken on a loaded machine is worse than none, so both are optional
flags supplied from a deliberate benchmark run.

Until the model is registered the hash on disk does not match the stored one, the Status
screen reports a mismatch and **inspection is stopped**. That is the intended behaviour
(T-23), not a setup step to work around.

| File | Change |
|---|---|
| `.env` | `INSPECTION_PROVIDER=real`, `MODEL_PATH`, optionally `MODEL_SHA256` to pin it |
| `requirements-real.txt` | already lists `onnxruntime`, `opencv-python-headless`, `scikit-image`; install it into `.venv-app` |
| `data/metrics/coverage.json` | replace with the metrics file the evaluation run emits |
| `model_version` table | written by `scripts/register_model.py` — not by hand |

Nothing in `app/services`, `app/routes`, `app/templates` or `app/static` changes. No
threshold changes either: those live in `app/profiles.py`, and re-tuning one there flows
through the seed into the database and onto the Materials screen with no other edit.

### Where `Profile` lives, and why it moved

The interface has to display the active thresholds (Phase 2 figure 8) and a test has to
prove it did not copy them (T-06). But `app/postprocess.py` imports cv2, numpy and
scikit-image, and the mock deployment installs none of them — so importing it there was
not an option, and a UI holding its own copy of the numbers would have started lying the
first time they were re-tuned.

So the dataclass lives in **`app/profiles.py`**, which is stdlib-only and imports
nothing. `app/postprocess.py` re-exports it, so `postprocess.Profile is profiles.Profile`
and there is still exactly one definition — asserted by
`tests/test_materials.py::test_postprocess_reexports_the_same_profile_object`.

## Testing the integration

| Case | Test |
|---|---|
| Missing model file | `tests/test_status.py::test_real_provider_refuses_a_missing_model` |
| Hash mismatch stops the run (T-23) | `test_real_provider_refuses_a_hash_mismatch` |
| Missing inference core | `test_real_provider_reports_a_missing_inference_core` |
| Status screen names the problem | `test_status_screen_reports_a_hash_mismatch_and_blocks` |
| Thresholds are not duplicated (T-06) | `tests/test_content_rules.py` |

`RealInspectionProvider` accepts an injected `inspector`, so the mapping can be tested
against a stub before the real weights exist.
