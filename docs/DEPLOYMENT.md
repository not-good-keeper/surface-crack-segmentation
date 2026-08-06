# Deployment

Two very different things go under this heading, and it is worth being blunt about the
difference before any of the commands below.

| | Station deployment | Vercel deployment |
|---|---|---|
| What it is | The real system | A shareable demo of the interface |
| Runs | CPU-only industrial PC on the line | Serverless functions |
| Provider | `real` — the ONNX pipeline | `mock` only |
| Network | None, by design (NFR-05) | Public internet |
| Database | Persists | Resets on every cold start |
| Purpose | Inspecting parts | Showing reviewers the screens |

**Vercel cannot run the real system**, and this is not a limitation to work around — it
is the point. The inference path deliberately opens no socket because factory images
must not leave the site; that requirement and a public serverless host are incompatible
by design. Use Vercel to show the interface. Ship the station on the industrial PC.

---

## 1. Vercel (demo)

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

### Deploy

```bash
# 1. Build the demo bundle (locally — this is the slow part, ~2 minutes)
python -m scripts.build_demo_bundle

# 2. Check the reported size. It should print ~24 MB and confirm it is under the
#    50 MB lambda limit set in vercel.json.

# 3. Commit the bundle. data/inspection.db and the generated images are deliberately
#    NOT gitignored for deployment — see the note in .gitignore.
git add -f data/inspection.db data/sources data/overlays data/batches
git commit -m "Build demo bundle"

# 4. Deploy
npm i -g vercel
vercel            # preview
vercel --prod     # production
```

`vercel.json` routes every request to `api/index.py`, which exposes the same FastAPI
application, and sets `INSPECTION_PROVIDER=mock`, `DEMO_MODE=true`, `SERVERLESS=true`.

### What works on Vercel

All six screens, region navigation, history filtering and pagination, CSV and JSON
exports, materials and thresholds, status checks and fault simulation, the demo
"Next inspection" control, and batch runs over the three bundled folders.

### What does not, and why

- **Writes do not persist.** Advancing the demo station or running a batch writes to
  `/tmp` on one instance. A different instance, or the same one after a cold start, sees
  the shipped database again. Fine for a demo; disqualifying for a station.
- **Instances do not share state.** Two people clicking around may be on different
  instances and see different results. There is no shared `/tmp`.
- **The first request after a cold start is slow** — it copies the database and imports
  NumPy and Pillow. Expect a second or two.
- **Batch runs must be small.** They run inside the request, against the function
  timeout (10 s on Hobby, 60 s on Pro). The bundled folders are 7, 5 and 4 images for
  this reason. A 500-image run will time out.
- **`INSPECTION_PROVIDER=real` will not work.** `onnxruntime` and `opencv-python-headless`
  push the bundle past the size limit, and pointing a factory pipeline at a public host
  contradicts NFR-05. The provider will refuse to start without a model file anyway.

### If the bundle is too large

```bash
python -m scripts.build_demo_bundle --live 40 --cap 360
```

`--cap` sets the longest image edge. At `--live 60 --cap 480` the bundle is about 24 MB.
The script warns above 40 MB.

### Alternatives worth considering

For anything beyond a demo, a container on a small VM keeps the application intact — a
persistent disk, real background threads, full-size images, and the option of running
the real provider on a machine you control. `python -m app.main` behind nginx is the
whole deployment. Fly.io, Railway and Render all take a Dockerfile; none of them require
the changes above.

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
