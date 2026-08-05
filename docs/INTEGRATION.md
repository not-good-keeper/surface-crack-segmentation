# Integrating the model into an application

What a UI needs in order to call this model correctly. The ONNX graph is **fixed** —
input and output shapes, channel order and normalisation do not change when weights are
retrained — so an app built against this contract keeps working when the final weights
land. Only the `.onnx` file is swapped.

## The short version

Use `app/inference.py`. Do not reimplement the pipeline.

```python
from app.inference import Inspector

insp = Inspector("data/export/smpslim_timm-mobilenetv3_small_100_v8a.onnx",
                 station_id="line-1-cam-A")

image_bytes = open("part.png", "rb").read()
image_bgr   = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

record, overlay_bgr, class_map = insp.inspect(image_bgr, image_bytes,
                                              product_id="batch-77/item-12",
                                              material="steel")
```

`record` is the `ARCHITECTURE.md` §8.1 JSON, `overlay_bgr` is a BGR image at the
**source resolution** (not 256×256), and `class_map` is an `(H,W)` int array of
0/1/2 for background/crack/scratch.

The reason to route through `Inspector` rather than calling onnxruntime directly is
that the numbers this project reports are the numbers this exact path produces. A
reimplementation that resizes differently, or applies argmax instead of the class
thresholds, is a different system with different accuracy, and no metric in the
benchmark would reveal the difference.

## The tensor contract, if you must call ONNX yourself

| | |
|---|---|
| input name | `input` |
| input shape | `(1, 3, 256, 256)` float32, batch axis dynamic |
| channel order | **RGB**, not BGR |
| preprocessing | `cv2.resize(..., (256,256), INTER_LINEAR)` → **input transform, see below** → `/255` → ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` → `CHW` |
| output name | `logits` |
| output shape | `(1, 3, 256, 256)` float32 |
| **channel meaning** | `0 = background`, `1 = crack`, `2 = scratch` |
| activation | none applied in-graph — the output is **raw logits**, apply softmax yourself |

### The input transform is part of the contract

The shipped weights are trained with a **bilateral filter** on the resized BGR frame,
before normalisation:

```python
img = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
img = cv2.bilateralFilter(img, 5, 50, 50)     # then /255, normalise, CHW
```

Skipping it does not throw — it quietly costs detection (0.962 → 0.949) and about half
the model's response on materials outside the training set.

You should not have to remember this. `bench/export_onnx.py` stamps the transform into
the graph's `metadata_props` under the key `prep`, and `Inspector` reads it back and
applies it. Swapping in a checkpoint trained with a different transform therefore
changes the preprocessing automatically. If you call onnxruntime directly, read it
yourself and honour it:

```python
spec = sess.get_modelmeta().custom_metadata_map.get("prep", "")
```

`bench/preprocess.py` holds the implementations; `bench/prep_sweep.py` is the sweep that
chose this one. An empty or missing `prep` means raw input — that is how every model
before v10 was trained, so old exports keep working.

Two more things bite people here. The model is trained on **plain resize, not letterbox** —
matching the training loader matters more than preserving aspect ratio, because
preprocessing that differs between training and deployment is a silent accuracy loss.
And the graph emits logits, not probabilities: feeding raw logits into a threshold
intended for probabilities will appear to work while being badly miscalibrated.

## The class decision is not argmax

`ARCHITECTURE.md` §7.1: compare each foreground class against its own floor; a pixel
below both is background; if both pass, the higher score wins.

```python
prob = softmax(logits)              # (3,H,W)
crack_pass   = prob[1] > 0.40       # Profile.crack_thresh
scratch_pass = prob[2] > 0.20       # Profile.scratch_thresh
```

Thresholds live in `app/postprocess.py::Profile` and were selected on the **validation**
split, never on a test split. Import them; do not copy the numbers into the UI, or the
two will drift the first time they are re-tuned.

Plain argmax is measurably worse at typing: on the frozen factory split it gives crack
class recall 0.460 against 0.491 for §7.1, at identical clDice. Argmax also has no
background floor at all — a pixel scoring 0.34 crack / 0.33 scratch / 0.33 background
comes out as a confident crack.

## Region geometry

After the class map: 8-connected components per class, then regions below
`min_area_px=24` or `min_skeleton_px=6` are dropped. Those floors come from the
clean-surface false-positive budget (NFR-03), not from tuning on test images.

Per region: `area_px`, `length_px` (medial-axis arc length, diagonal steps counted as
√2 — summing skeleton pixels would understate a diagonal defect by ~41 %), and
`max_width_px` (twice the maximum distance-transform value along the skeleton).

**One known limitation to surface in the UI**: `max_width_px` is bounded by the narrow
dimension of a region. A local bulge in a thin crack does not register as a large
maximum width, because the inscribed circle at that point is still constrained by the
crack's narrowness. It measures the widest *inscribable* point, not the widest visual
extent.

## What to show the operator, and what not to

Per `ARCHITECTURE.md` §9 and NFR-06:

- **Do** show the overlay, the region list, and the geometry.
- **Do not** show a confidence percentage. The scores are uncalibrated; rendering
  `0.87` as "87 % confident" states a probability the model has never been validated to
  produce.
- **Do not** show an accept/reject verdict. That requires customer tolerances the system
  does not have, and it is explicitly out of scope in `REQUIREMENTS.md`.
- **Do** distinguish "clean" from "failed to process". A defect-free product is recorded
  with `"empty": true` and an empty region list; an unreadable image is a structured
  acquisition failure. Collapsing those two into one "no defects" state loses the
  difference between a good part and a broken camera.

## Honest limits worth surfacing in the UI

Every one of these is measured, not hedging:

| Material | Status |
|---|---|
| Steel | Supported. Best class typing (0.702), weakest geometry (clDice 0.529) — steel cracks are the thinnest in the corpus at ~4 px |
| Plastic | **One product only.** All 936 real masks are PVC pipe; a moulded casing is not covered |
| Ceramic | Thin coverage (120 training masks); class typing 0.315 |
| Epoxy | **Zero training masks.** Detection works, typing does not (0.279) |
| Glass | **Not supported.** 12 masks in the whole corpus |
| Non-steel metal | **Not supported.** Zero masks |

Scratch evidence is 624/724 steel, so a `scratch` label on ceramic, plastic or glass is
transfer, not evidence.

**Crack/scratch typing is the open defect**: the model detects 96 % of defects but
assigns the correct *type* to only ~49 % of crack pixels, biased toward scratch. If the
UI's value depends on the type label being right, treat it as provisional and show the
mask as the primary output.

## Latency

26 ms per 256×256 image, single-threaded desktop CPU, model 1.43 M parameters / 5.8 MB
float32. Single-threaded deliberately: the deployment target is a CPU-only industrial PC
also running line software, and a benchmark taken with every core free is not a number
anyone can plan against.

**No ARM or phone measurement exists**, and the corresponding requirement was withdrawn
rather than carried as an untested claim.

## Offline by construction

Nothing in `app/inference.py` opens a socket: onnxruntime on CPU, a local weights file,
no client library that could phone home (NFR-05). Keep it that way in the UI layer —
factory IP is the reason this constraint exists.
