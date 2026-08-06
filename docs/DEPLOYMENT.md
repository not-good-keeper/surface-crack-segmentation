# Deployment

Two supported paths. Both run the real ONNX pipeline; what differs is where they run
and what that costs.

| | Station | Container (hosted demo) |
|---|---|---|
| Section | §2 below | §0 below |
| Provider | `real` | `real` |
| Inference | Local ONNX, CPU | Local ONNX, CPU — same `Inspector` |
| Network | None, by design (NFR-05) | Public internet: images travel to reach it |
| Filesystem | Local disk, persists | Container filesystem, reset on redeploy |
| Purpose | Inspecting parts | Showing the working system end to end |

The station is the real system: one persistent local FastAPI process with a local ONNX
model, SQLite database and local media directories. Use
`uvicorn app.main:app --host 127.0.0.1 --port 8000`; see `LOCAL_SETUP.md` and the
README.

The container path (§0) runs the *same* application with the *same* model — uploads and
browser-camera frames go through `app/inference.py::Inspector` exactly as they do on the
station. The honest difference is not the code, it is the deployment: NFR-05's offline
guarantee is a property of running on the line with no network, and a public host is by
definition not that. The inference path still opens no outbound socket. **Do not point a
production line at a public deployment.**

§1 below, the Vercel material, is retained only as an architectural record of why the
old read-only/`/tmp` split existed and why serverless was abandoned; it is not a
supported procedure, and `tests/test_deployment.py` asserts no `vercel.json` or
`api/index.py` has come back.

---

## 0. Container hosting (hosted demo)

`Dockerfile` builds a full real-mode image: `INSPECTION_PROVIDER=real`, the exported
graph at `data/export/model.onnx`, and the whole inference stack from
`requirements.txt` (`onnxruntime`, `opencv-python-headless`, `scikit-image`).

Two build-time steps, in order:

1. **Seed** (`scripts.seed_db --live 60`) generates the demonstration history the
   History and Analytics screens read. It runs the **mock** provider deliberately —
   those rows are procedurally generated sample data, not measurements — and it happens
   at build time because seeding takes minutes and a container should answer its first
   request immediately.
2. **Register** (`scripts.register_model`) makes the real ONNX the active
   `model_version`, computing the hash from the file itself. That is why
   `MODEL_SHA256` is left empty: the database row becomes the reference that
   `status_service.model_check` verifies the file against, so a swapped model still
   stops inspection (T-23) without pinning a literal in two places.

So the pre-loaded history is generated sample data, while **every new inspection —
upload, camera capture, live inspection, batch — is real model output.**

Two system libraries are installed because `python:*-slim` lacks them: `libgomp1`
(onnxruntime's OpenMP runtime; without it the import fails with a bare
`libgomp.so.1: cannot open shared object file`) and `libglib2.0-0` (opencv's remaining
shared dependency even in the headless build).

### The model is committed on this branch

Render builds from git, so the graph has to be in the repository — there is no build
step that could fetch it. It is committed **on the `hostable` branch only**; `main`
does not carry it, because a station installs its own model file and records the hash
(README, "Replacing the model").

The file is self-contained: `torch.onnx.export` split the weights into a sidecar
`.onnx.data`, which was merged back in. That matters for more than tidiness —
`model_check` hashes the `.onnx` alone, so weights in a separate file would sit
entirely outside the integrity check while Status still reported `verified`.

**Licence caveat.** The training corpus mixes CC0, CC-BY, CC-BY-NC and research-only
terms (`docs/ATTRIBUTION.md`). Publishing these weights is redistribution. Committing
them here was a deliberate decision for a portfolio demo — not a finding that the terms
permit it. Re-check before any commercial use or wider distribution.

```bash
docker build -t vision404-demo .
docker run -p 8000:8000 vision404-demo
# or: docker compose up --build
```

The container reads `$PORT` if the host sets it (Render, Railway, Fly.io and Cloud Run
all do this); it falls back to 8000. `/healthz` is wired up as the image's
`HEALTHCHECK`.

### Deploy to Render (free tier)

`render.yaml` at the repository root is a
[Render Blueprint](https://render.com/docs/blueprint-spec): Render reads it
automatically once the repository is connected, so no manual service configuration is
needed.

1. Push this branch to a GitHub/GitLab repository Render can see (a fork is fine).
2. In the Render dashboard: **New +** -> **Blueprint**, pick the repository and branch.
3. Render finds `render.yaml`, builds `Dockerfile` on the free web-service plan, and
   sets `INSPECTION_PROVIDER=real` / `DEMO_MODE=true`. `$PORT` and the `/healthz`
   health check are wired up already — nothing else to configure.
4. First build takes several minutes (installing the inference stack, then seeding the
   demonstration history). After that, every deploy repeats the same two build steps.

**The camera needs HTTPS, and Render provides it.** Browsers only expose
`getUserMedia` in a secure context, so Capture and Live work on a `*.onrender.com` URL
exactly as they do on `localhost`. Serving this over plain HTTP on a custom domain
would silently break the camera while everything else kept working.

Render's free plan spins the instance down after 15 minutes idle; the next request
wakes it back up and pays for loading the ONNX session, so expect a few seconds.

**Memory.** The free plan caps at 512 MB. Measured locally: 71 MB with the ONNX session
loaded, 98 MB after a 1024×1024 inference, 127 MB after 2048×2048 — comfortable, but
`OMP_NUM_THREADS=1` is set to keep it predictable on a shared instance (and it matches
how latency was benchmarked: one CPU thread).

Railway and Fly.io also build straight from `Dockerfile` with no extra configuration
(connect the repository and pick "Docker" as the environment on Railway; `fly launch
--dockerfile Dockerfile` on Fly.io) — `render.yaml` is Render-specific and simply
unused on either of them.

### What works

Everything, with real model output:

- **Upload** a PNG/JPEG on Capture → real ONNX inference, real geometry, real overlay.
- **Camera capture** and **live inspection** → same pipeline, same `POST /api/inspections`.
- **Batch runs** over the bundled folders.
- **Analytics** (`/analytics`, `/analytics/{batch_run_id}`), History, Regions,
  Materials, Status, CSV/JSON exports.

Verified locally against the built image's configuration: `/api/status` reports
`STATION OK` with the model check `verified`, and an uploaded 512×512 probe returned
`regions_found` with a 893 px crack region measured at source resolution.

### What does not, and why

- **The demo "Next inspection" control is absent.** It only advances the *mock*
  station; `live.html` and `status.html` gate it on `demo_mode and provider == "mock"`,
  so with the real provider it is not rendered at all rather than rendered broken.
  Upload or the camera is how you produce a new inspection here.
- **Writes do not persist across a redeploy.** A new build starts from the build-time
  seed again; inspections recorded in between are gone. Fine for a demo, disqualifying
  for a station.
- **No authentication.** Same as the local server: anyone with the URL can upload an
  image and read the history. Don't put anything sensitive behind it.
- **It is still not the station.** Images reach this container over the public
  internet. That is the one guarantee a hosted deployment cannot make (NFR-05), and no
  amount of configuration changes it.

---

## 1. Vercel (retired, historical record only)

**Vercel cannot run the real system**, and that was never a limitation to work around —
it was the point. The inference path deliberately opens no socket because factory
images must not leave the site; that requirement and a public serverless host are
incompatible by design. Vercel was used to show the interface only; the station always
shipped separately, on the industrial PC (§2).

The project no longer deploys to Vercel — §0 (Docker, above) replaced it, on a host
with a normal filesystem instead of the read-only/`/tmp` split serverless forced. What
follows is kept only because it explains *why* that split existed; none of it is
runnable today (`tests/test_deployment.py` asserts `vercel.json` and `api/index.py`
stay gone).

### Why the application needed changes at all

A serverless function has a **read-only filesystem apart from `/tmp`**, is **ephemeral**,
and has a **short execution limit**. This application writes a SQLite database, generates
images, and takes about three minutes to seed. Three changes make it work:

| Problem | What the code does |
|---|---|
| Filesystem is read-only | Two data roots. `bundled_data_dir` (shipped, read-only) and `runtime_data_dir` (`/tmp`, writable). Set by `SERVERLESS=true` / the `VERCEL` env var. |
| Seeding takes minutes, a cold start has seconds | The database is seeded **at build time** by `scripts/build_demo_bundle.py` and shipped. `lifespan` copies it into `/tmp` on the first request and never seeds on demand. |
| A database built on one machine, read on another | Image paths are stored **relative to the data root**, so they resolve wherever the bundle lands. |
| Background threads are frozen once a response is sent | Batch runs go synchronous when `is_serverless` — `settings.run_batches_synchronously`. |

None of this touches the screens, the schema, the provider boundary or the tests.
`scripts/build_demo_bundle.py` (the bundle builder referenced above) still exists in
the repository as of this writing but has no current caller — §0 seeds the container
at build time instead and needs no committed bundle.

Constraints this approach ran into, which is why §0 does not: writes did not persist
past a cold start; two visitors could land on different instances with different
state; a cold start was slow (copying the database, importing NumPy and Pillow); batch
runs had to stay tiny to fit the function timeout; and `INSPECTION_PROVIDER=real`
could never work — the bundle size limit and NFR-05 both forbid it, same as §0.

---

## 2. Station deployment (the real system)

The target is a **CPU-only industrial PC that is also running line software**. One unit
per station; units share nothing at runtime, so stations do not contend.

```bash
git clone <repo> /opt/vision404
cd /opt/vision404
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install onnxruntime opencv-python-headless scikit-image

cp .env.example .env
```

```env
INSPECTION_PROVIDER=real
MODEL_PATH=/opt/vision404/data/export/smpslim_timm-mobilenetv3_small_100_v8a.onnx
MODEL_SHA256=<sha256 of that file>
STATION_ID=line-1-cam-A
DEMO_MODE=false
MIN_FREE_DISK_GB=20
```

`DEMO_MODE=false` removes the "Next inspection" control and the fault simulation: on a
line, the line advances the station.

Run under systemd:

```ini
[Unit]
Description=Surface Defect Inspection (line-1-cam-A)
After=network.target

[Service]
WorkingDirectory=/opt/vision404
ExecStart=/opt/vision404/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
User=inspection

[Install]
WantedBy=multi-user.target
```

Bind to `127.0.0.1` unless the interface genuinely needs to be reachable from another
machine on the line network. There is no login and no user accounts — the system cannot
attribute an action to a person, which is a real limitation and a deliberate consequence
of dropping the verdict.

### Verify after install

```bash
python -m pytest                       # 250 tests
curl localhost:8000/api/status         # every check should report ok
```

Confirm on the Status screen that **Model file / hash** reads `verified`. If it reports a
mismatch, inspection is stopped by design — the system will not produce results from a
file it cannot identify (T-23).

### Offline check (T-20)

Disable the network and run a live inspection and a batch run. Both must finish. Nothing
in the inference path or the interface opens a socket: onnxruntime on CPU, a local
weights file, locally served CSS and JavaScript, no CDN and no web font.

### Ongoing

- **Disk** is a status check. Below `MIN_FREE_DISK_GB` the run stops rather than dropping
  records silently. Retention (48 h for clean images, 90 days where regions were found)
  is a scheduled job — see DATABASE.md; it is not automated in this build.
- **Backups**: copy `data/inspection.db` and `data/records.jsonl` on a schedule from a
  separate machine. The line never waits for it.
- **New weights**: swap the `.onnx`, record the new hash in `MODEL_SHA256` and in
  `model_version`, restart. No code change unless the input or output shape changes.
