# Inspection app

Local, offline prototype of the inspection station described in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md). One segmentation model, deterministic
post-processing, and a structured record per product.

## Run

```bash
pip install onnxruntime opencv-python pandas scikit-image
python app/batch.py --model data/export/<name>.onnx --images <dir> --out data/inspect
```

The batch CLI is the supported interface. `app.py` is a Streamlit view kept only as a
runnable demo — the real UI is being built separately, so nothing depends on it and
deleting it costs nothing.

Needs an exported model in `data/export/`:

```bash
python bench/export_onnx.py --models smpslim_timm-mobilenetv3_small_100 \
    --classes 3 --tag v10 --prep bilateral \
    --weights data/bench/<checkpoint>.pt --allow-busy
```

`--prep` records the input transform the weights were trained with inside the ONNX
metadata. `Inspector` reads it back and applies it, so the app cannot drift from the
training pipeline — see architecture §5.1. An export without it means raw input, which
is how every model before v10 was trained.

A binary checkpoint also loads, and the app says so on screen — it can only ever report
cracks, so every scratch it finds will be labelled `crack`.

## Batch

```bash
python app/batch.py --model data/export/<name>.onnx --images <dir> --out data/inspect
```

Produces `report.csv` (one row per region, one row per clean product), `records.json`,
overlays, and an appended `audit.jsonl`.

## Layout

| file | role |
|---|---|
| `postprocess.py` | architecture §7: thresholds → components → skeleton length, distance-transform width → overlay. Also the §8.1 record |
| `inference.py` | ONNX session, input pipeline identical to `bench/data.py`, single-threaded CPU |
| `app.py` | Streamlit demo view. Contains no measurement of its own, and nothing imports it |
| `batch.py` | directory → CSV/JSON/audit. Same `Inspector`, same `Profile` |
| `test_pipeline.py` | geometry asserted against shapes with known dimensions; app-vs-batch equivalence |

The app and the batch CLI import the same two modules deliberately. The failure worth
preventing is an operator approving a part on screen while the QC report records
something else, and that becomes possible the moment two copies of "measure the width"
exist.

## What it does not do

No pass/fail verdict and no confidence percentage. Per architecture §9 the system takes
on the *search* load and leaves the *judgement* to a person; acceptance tolerances are
per-customer, and the model's scores are not calibrated, so a displayed "97 % sure"
would be read as meaning something it does not.

Geometry is reported in **pixels**. Millimetres require a calibrated scale reference at
the inspection plane (architecture §5).

```bash
python app/test_pipeline.py
```
