# Industrial Surface-Defect Inspection System — Interface

**Team VISION 404** · Phase 2 application layer

A local web interface for the surface-defect inspection pipeline. It shows the operator
what was found, stores every inspection so it can be explained later, and exports the
results. It runs today against a deterministic mock provider, and switches to the real
ONNX pipeline with one configuration change.

---

## What this is, and what it deliberately is not

The interface reports **what was found** and shows the evidence. It does not decide
whether a part is acceptable.

| It does | It does not |
|---|---|
| Show the overlay at source resolution, the region list and the geometry | Show a confidence percentage — the model's scores are not calibrated, and rendering `0.87` as "87 % confident" states a probability the model has never been validated to produce |
| Keep `clean` and `could not process` as different outcomes | Show an accept/reject verdict — that needs customer tolerances the system does not have |
| Mark the class label provisional wherever it appears | Claim the class label is reliable — typing is about 49 % correct on crack pixels |
| Read thresholds from `app/postprocess.py` at start-up | Hold its own copy of the thresholds |

## Current status: mock mode

`INSPECTION_PROVIDER=mock` is the default and the application **starts with no ONNX
file present**. Results on screen are generated, not measured, and every screen says so.
The mock exists so the interface, database, exports and tests could be finished and
reviewed before the final weights land — and so that when they do, only the provider
changes.

---

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Runtime only (no test tooling):

```bash
pip install -r requirements.txt
```

## Seed the database

The first run seeds itself automatically in mock mode. To do it explicitly:

```bash
python -m scripts.seed_db                          # ~140 inspections, 3 batch runs
python -m scripts.seed_db --anchor 2026-08-06      # fixed dates, byte-reproducible
python -m scripts.seed_db --live 40 --no-batches   # smaller and faster
```

Takes two to three minutes: it generates every demo image procedurally.

## Reset

```bash
python -m scripts.reset_db                         # drop, rebuild, reseed
python -m scripts.reset_db --anchor 2026-08-06
```

## Run

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The application never opens an outbound connection: no
CDN, no web font, no telemetry. It works with the network cable out.

## Test

```bash
pytest                              # everything (250 tests, ~90 s)
pytest tests/test_ui.py             # browser tests only
pytest -m "not ui"                  # skip the browser
ruff check .                        # lint
```

Browser tests need Chromium:

```bash
python -m playwright install chromium
```

They skip automatically when it is not installed.

## Screenshots

```bash
python -m scripts.capture_screenshots
```

Writes 20 PNGs to `docs/screenshots/` — every screen at 1440 px and 390 px, plus the
filtered, empty, clean, failure and station-fault states.

---

## Configuration

Everything is environment-driven; see `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `INSPECTION_PROVIDER` | `mock` | `mock` or `real` |
| `DEMO_MODE` | `true` | Enables the "Next inspection" control and fault simulation |
| `MODEL_PATH` | `data/export/model.onnx` | The exported ONNX file |
| `MODEL_SHA256` | *(blank)* | Expected hash. A mismatch stops inspection |
| `STATION_ID` | `line-1-cam-A` | Station this instance serves |
| `DATABASE_PATH` | `data/inspection.db` | SQLite file |
| `BATCH_ROOT` | `data/batches` | Batch folders. Paths outside it are refused |
| `MIN_FREE_DISK_GB` | `5` | Below this the disk check fails and the run stops |
| `MOCK_SEED` | `404` | Makes the mock reproducible |

## Switching to the real model

```bash
INSPECTION_PROVIDER=real
MODEL_PATH=data/export/smpslim_timm-mobilenetv3_small_100_v8a.onnx
MODEL_SHA256=<sha256 of that file>
```

`app/inference.py` (the `Inspector` class) must be importable. The adapter verifies the
hash before loading and refuses to start on a mismatch. Full detail in
[MODEL_INTEGRATION.md](MODEL_INTEGRATION.md).

## Deploying

For a shareable demo: [DEPLOYMENT.md](DEPLOYMENT.md) covers Vercel, including what a
serverless host can and cannot do with this application. **A serverless deployment is a
demonstration of the interface, not a station deployment** — the station is a CPU-only
industrial PC, offline, with a database that persists.

---

## Screens

| Screen | Route | Purpose |
|---|---|---|
| Live Inspection | `/live` | The result for the item at the station |
| Region Detail | `/regions` | Each region enlarged, with its measurements |
| Batch Run and Report | `/batch` | Run a folder, reconcile the totals, export |
| Inspection History | `/history` | Filter and page through past inspections |
| Materials and Thresholds | `/materials` | Measured coverage, active thresholds, model |
| System Status | `/status` | Camera, model hash, database, disk |

## Layout

```
app/
├── main.py, config.py, dependencies.py, postprocess.py
├── providers/      base.py  mock_provider.py  real_provider.py  mock_assets.py
├── services/       inspection  batch  history  material  status  export
├── repositories/   inspection  material  model  batch
├── database/       connection.py  schema.sql  migrations.py  seed.py
├── schemas/        typed request and response models
├── routes/         pages.py  api_*.py  media.py
├── templates/      six screens + partials
└── static/         css/  js/  icons/
api/index.py        serverless entry point
scripts/            seed_db  reset_db  build_demo_bundle  capture_screenshots
tests/              250 tests
data/               database, generated images, metrics, batch folders
```

---

## Known limitations

Measured, not hedged. All of these are visible in the interface as well as here.

- **Class typing is the open defect.** Detection is around 96 %, but the correct *type*
  is assigned to only about 49 % of crack pixels, biased towards scratch. The mask is
  the primary output; the class label is marked provisional everywhere it appears.
- **Material coverage is uneven.** Steel is supported. Plastic is one product only (all
  936 masks are PVC pipe). Ceramic has thin coverage. Epoxy detects but cannot be typed.
  Glass and non-steel metal are not supported. A scratch label on anything but steel is
  transfer, not evidence.
- **Maximum width is the widest *inscribable* point**, not the widest visual extent. A
  local bulge in a thin crack does not register.
- **Only the inference stage of latency is measured**: 26 ms on one desktop CPU thread.
  No end-to-end figure is quoted because none has been measured.
- **No ARM or phone measurement exists.** The requirement was withdrawn rather than
  carried as an untested claim.
- **No int8 model.** The 5.8 MB float32 export is within the original 6 MB limit.
- **No user accounts, so no attribution.** The system cannot tell you who did what.
- **Mock results are not model measurements.** Nothing produced in mock mode should be
  quoted as accuracy.
- **Two datasets are non-commercial** (KolektorSDD2, MVTec AD, both CC BY-NC-SA 4.0).
  Fine for the prototype, blocking for a product.
- **Subsurface flaws are not detected.** This is an optical system and does not replace
  ultrasonic, radiographic or penetrant testing.

## Documentation

| File | Contents |
|---|---|
| [UI_IMPLEMENTATION_NOTES.md](UI_IMPLEMENTATION_NOTES.md) | Screens, routes, templates, responsive behaviour, accessibility, mock data |
| [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) | The provider boundary, the ONNX contract, switching to real |
| [DATABASE.md](DATABASE.md) | Schema, relationships, indexes, seeding, retention, backup |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Vercel deployment and its limits |
