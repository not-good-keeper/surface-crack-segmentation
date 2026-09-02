# Conveyor-line surface defect segmentation

Pixel-level crack and scratch segmentation for finished factory products passing under a
fixed camera on a conveyor. Built for Indian MSME lines — steel fabrication and casting,
ceramic tile, plastic moulding — where a commercial vision system is out of reach.

The output is a mask, a defect type and per-region geometry. It is **not** an
accept/reject verdict: that needs customer tolerances the system does not have, and
over-rejection is the failure this project exists to avoid.

## Authorship

A two-person hackathon project (Team VISION 404). The split of work:

| Area | Author |
|---|---|
| Dataset pipeline, synthetic compositor, corpus QA and splits — `dataset/`, `defectforge/` | Nethi Kushala Kumar |
| Model training, metrics, ablations, evaluation and ONNX export — `bench/`, `experiments/` | Nethi Kushala Kumar |
| Inference core and Phase 2 web application — `app/`, `scripts/`, `tests/` | Nethi Kushala Kumar |
| Analytics dashboard, container hosting and deployment | Nethi Kushala Kumar |
| Phase 1/2/3 project documents — `CV_Hackathon - Phase *.pdf` | Kanishk Rungta |

The commit history here is the original authored record — the system was built across
20 commits between 01:27 and 15:51 on 6 August 2026, and each one is scoped to a single
decision. The team repository is
[Kanishk-Rungta/INDUSTRIAL-SURFACE-DEFECT-INSPECTION-SYSTEM](https://github.com/Kanishk-Rungta/INDUSTRIAL-SURFACE-DEFECT-INSPECTION-SYSTEM),
where the same code was imported as a single squashed commit; this repository preserves
the per-change history behind it.

## Current results

Three seeds on frozen splits v7 (`c0fde17c96749567`), MobileNetV3-Small + slim U-Net,
1.43 M parameters, 26 ms per image on one desktop CPU thread.

| Split | clDice | IoU | Detection |
|---|---|---|---|
| `test_factory` (steel, ceramic, epoxy, plastic, glass) | 0.6771 ±0.010 | 0.4646 ±0.008 | **0.9615 ±0.010** |
| `test_factory_scratch` (n=31) | 0.8744 ±0.004 | 0.6759 ±0.009 | 1.0000 |
| `test_unseen_material` (wood, never trained on) | 0.7259 ±0.013 | 0.3413 ±0.007 | 0.9556 |
| `test_negatives` (clean surfaces) | — | — | **false-positive area 0.0046 ±0.001** |

False-positive area meets the 0.5 % requirement. An all-background model scores 95.6 %
pixel accuracy on `test_factory` and 0.0 clDice, which is why accuracy is never quoted
here.

**The open defect**: crack/scratch *typing* is only 0.49 on the headline split. Defects
are found reliably; the type label is right for about half of crack pixels, biased toward
scratch. Class recall tracks per-material training data almost monotonically — steel
0.702 (1,047 masks), plastic 0.657 (749), ceramic 0.315 (120), epoxy 0.279 (**zero**) —
so this is a data gap, not a threshold or an architecture problem. It is reported rather
than smoothed over, because the class-agnostic headline metrics do not reveal it.

## Layout

| Path | What it holds |
|---|---|
| `dataset/` | Fetch, adapt, normalise, split, QA. Regenerates the corpus from source |
| `bench/` | Training, metrics, model zoo, evaluation and export |
| `app/` | **Both layers.** Inference core (`inference.py`, `postprocess.py`, `profiles.py`, `batch.py`) and the Phase 2 web application (`main.py`, `routes/`, `services/`, `repositories/`, `providers/`, `templates/`, `static/`) |
| `Dockerfile`, `render.yaml` | Container entry point for the public demo — the same ASGI app in mock mode |
| `scripts/` | Seed, reset, register a model, build the deployable demo bundle |
| `tests/` | Application test suite. The pipeline's own tests stay in `app/test_pipeline.py` |
| `defectforge/` | Synthetic defect compositor |
| `experiments/` | One script per stored result — see `experiments/README.md` |
| `docs/` | Requirements, architecture, dataset provenance, decisions, results index |

## Two environments, on purpose

The training stack pins torch on **Python 3.9**; the web application needs **3.11+**
(`pydantic-settings`, `datetime.UTC`). One interpreter cannot serve both, so there are two:

| Environment | Python | For | Holds |
|---|---|---|---|
| `.venv` | 3.9 | Training, benchmarking, export | torch, timm, albumentations, cv2, onnxruntime |
| `.venv-app` | 3.13 | The web application and its tests | fastapi, pydantic, Pillow — plus cv2/onnxruntime/scikit-image for real mode |

`app/inference.py`, `app/postprocess.py`, `app/profiles.py` and `app/batch.py` are
imported by both, so they are kept 3.9-compatible. `pyproject.toml` records why ruff is
not allowed to "modernise" them.

## Running the interface

```bash
.venv-app/Scripts/python.exe -m pip install -r requirements-dev.txt
cp .env.example .env
.venv-app/Scripts/python.exe -m app.main            # http://127.0.0.1:8000
.venv-app/Scripts/python.exe -m pytest tests
```

It starts in **mock mode** with no model file: every screen works, and each says the
results are generated rather than measured. To inspect for real:

```bash
# 1. record the exported model, so the Status screen can identify it
.venv-app/Scripts/python.exe -m scripts.register_model \
    --model data/export/<file>.onnx --version v13 --params 1430000

# 2. switch the provider
INSPECTION_PROVIDER=real MODEL_PATH=data/export/<file>.onnx \
    .venv-app/Scripts/python.exe -m app.main
```

Until `register_model` is run, the stored model hash does not match the file and
inspection is **stopped** on purpose — a system that quietly accepted an unidentified
model file would produce results it could not attribute to anything.

Screens: Live, **Capture**, Regions, Batch, History, Materials, Status. Capture is the
only one not in the Phase 2 wireframes; it takes a frame from the device camera or a
photo from disk and runs it through the same provider, record mapping and database
writer as a station inspection. Deployment is covered in `docs/DEPLOYMENT.md` — the
short version is that the container runs the mock demo and the station runs the real pipeline,
and those are different things by design, not by limitation.

## Reproducing the corpus

Images are not in this repository. They are regenerable, and the licences forbid
redistributing several of them.

```bash
python dataset/fetch.py            # sources and checksums in dataset/sources.yaml
python dataset/index.py
python dataset/normalize.py
python dataset/split.py            # writes data/splits.json + manifest_split.csv
python dataset/qa.py --strict      # leakage assertions; blocks training claims on failure
```

## Training and evaluation

```bash
bash run_v9.sh                                            # 3 seeds, current recipe
python bench/summarize.py     --tag V9                    # mean ± std across seeds
python bench/per_material.py  --tag V9_22                 # headline split, by material
python bench/final_eval.py    --tag V9_22                 # every split, no subsampling
python bench/export_onnx.py --models smpslim_timm-mobilenetv3_small_100 \
       --classes 3 --tag v9 --weights data/bench/..._V9_22.pt
```

Export verifies itself against the torch model (logit drift and per-pixel argmax
agreement) and refuses to measure latency on a busy machine.

## Using the model in an application

See **`docs/INTEGRATION.md`** for the tensor contract, the class-decision rule and the
limits worth surfacing to an operator. The short version: use `app/inference.py`, don't
reimplement the pipeline, and don't show a confidence percentage — the scores are
uncalibrated.

## Documentation

### Documents, in reading order

Requirement IDs (`FR-nn`, `NFR-nn`) are the ones defined in the submitted **Phase 1**
report; test IDs (`T-nn`) are **Phase 2**'s. Nothing here is renumbered — a document that
invents its own IDs forces a reader holding the report to build the mapping themselves.

| # | File | Contents |
|---|---|---|
| 1 | `docs/REQUIREMENTS.md` | Every FR and NFR with its status: met, partial, interface, withdrawn or not met — and where each is implemented |
| 2 | `docs/ARCHITECTURE.md` | Pipeline, input preparation (§5.1), model, post-processing, result schema, status (§10.3), modularity and scale (§11) |
| 3 | `docs/DATASET.md` | Reconciliation with the Phase 1 dataset plan (§0), the corpus, every source and why it was accepted or rejected, synthetic policy, what was not delivered (§8), quality gates (§9) |
| 4 | `docs/TEST_PLAN.md` | T-01…T-30 with per-case status, traceability to requirements, and an honest automation count |
| 5 | `docs/DECISIONS.md` | 21 decisions with the measurement behind each, including two corrections to earlier readings |
| 6 | `docs/RESULTS.md` | Which stored run supports which claim |
| 7 | `docs/INTEGRATION.md` | How to call the model from an app: tensor contract, class decision, geometry, limits |
| 8 | `docs/ATTRIBUTION.md` | Licences. The corpus mixes CC0, CC-BY, CC-BY-NC and research-only terms |

### The Phase 2 application

| File | Contents |
|---|---|
| `docs/UI_README.md` | What the interface shows and refuses to show, install, seed, run, test |
| `docs/UI_IMPLEMENTATION_NOTES.md` | Screens, routes, templates, responsive behaviour, accessibility, components |
| `docs/MODEL_INTEGRATION.md` | The provider boundary, mock vs real, and what must never be reimplemented in the UI |
| `docs/DATABASE.md` | Schema, relationships, indexes, seed data, retention |
| `docs/DEPLOYMENT.md` | Station deployment vs the container demo, and why they are not the same thing |

Known gaps and unmet claims are recorded where the work is, not in a separate list:
open items in `ARCHITECTURE.md` §10.3, withdrawn requirements in `REQUIREMENTS.md`,
per-material coverage limits in `INTEGRATION.md`.

## Licence

**CC BY-NC-SA 4.0** — see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Use, adapt and
redistribute non-commercially, with credit; derivatives carry the same licence.

The terms are set by the data, not by preference. The trained model derives from sources
including non-commercial and research-only data (MVTec, KolektorSDD2, the casting set),
and a model trained on NC data is not automatically free of the NC condition, so the
restriction reaches `data/export/model.onnx` and anything built from it. Any release
must re-check `docs/ATTRIBUTION.md` per source — several are research-only and some were
deliberately excluded from redistribution.
