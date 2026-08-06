# Deployment

Two supported paths, and they are not interchangeable.

| | Station (real) | Container (demo) |
|---|---|---|
| Section | §2 below | §0 below |
| Provider | `real` only | `mock` only |
| Network | None, by design (NFR-05) | Public internet |
| Filesystem | Local disk, persists | Container filesystem, persists until redeploy |
| Purpose | Inspecting parts | Showing reviewers the screens, including Analytics |

The station is the real system: one persistent local FastAPI process with a local
ONNX model, SQLite database and local media directories. Use
`uvicorn app.main:app --host 127.0.0.1 --port 8000`; see `LOCAL_SETUP.md` and the
README. It has no remote inference endpoint, ever.

The container path (§0) is new: a Dockerfile that runs the same application in mock
mode for a public demo, on a normal host with a normal (if ephemeral-across-redeploys)
filesystem — not the read-only/`/tmp` split the retired Vercel path needed. §1 below,
the Vercel material, is retained only as an architectural record of why that split
existed and why serverless was abandoned for it; it is not a supported deployment
procedure and `tests/test_deployment.py` asserts no `vercel.json` or `api/index.py`
has come back.

---

## 0. Container hosting (demo)

`Dockerfile` builds a mock-mode image: `INSPECTION_PROVIDER=mock`, `DEMO_MODE=true`,
demonstration data generated procedurally by `app/providers/mock_assets.py` and seeded
**at image build time** (`RUN python -m scripts.seed_db`), so the container answers its
first request immediately instead of spending the ~2 minutes a fresh seed takes. The
image installs only what mock mode imports — `fastapi`, `uvicorn`, `jinja2`, `pydantic`,
`pydantic-settings`, `python-multipart`, `numpy`, `pillow` — not `onnxruntime`,
`opencv-python-headless` or `scikit-image`, which a public host must never run anyway
(see the table above and NFR-05).

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
   sets `INSPECTION_PROVIDER=mock` / `DEMO_MODE=true`. `$PORT` and the `/healthz`
   health check are wired up already — nothing else to configure.
4. First build takes a few minutes (the image seeds its demo data during the build,
   per above). After that, every deploy repeats the same build-time seed.

Render's free plan spins the instance down after 15 minutes idle; the next request
wakes it back up, a few seconds slower than usual. That is a demo-tier limitation, not
a bug in this application — see "What does not, and why" below.

Railway and Fly.io also build straight from `Dockerfile` with no extra configuration
(connect the repository and pick "Docker" as the environment on Railway; `fly launch
--dockerfile Dockerfile` on Fly.io) if a different host is preferred later — `render.yaml`
is Render-specific and simply unused on either of them.

### What works

Every screen, including **Analytics** (`/analytics`, `/analytics/{batch_run_id}`) —
the per-session dashboards are pure server-rendered SVG computed from the seeded rows,
so they need nothing beyond what mock mode already provides. Region navigation,
history filtering and pagination, CSV/JSON exports, materials and thresholds, status
checks and fault simulation, the demo "Next inspection" control, and batch runs over
the bundled folders.

### What does not, and why

- **`INSPECTION_PROVIDER=real` is refused in spirit, not just in practice.** The image
  never installs `onnxruntime`/`opencv-python-headless`/`scikit-image`, so real mode
  would fail to import even if the environment variable were set. This is deliberate:
  factory images must not leave the site (NFR-05), and a public container is not the
  site.
- **Writes do not persist across a redeploy.** A new image build starts from the
  build-time seed again; whatever an operator clicked in the meantime is gone. Fine for
  a demo, disqualifying for a station — same caveat the retired Vercel path had, for a
  different reason (that one couldn't persist writes at all, even between requests).
- **No authentication.** Same as the local server: don't put anything behind this that
  needs access control without adding it first.

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
