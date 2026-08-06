# Container image running the FULL application - real ONNX inference included.
#
# This is the whole system, not a mock of it: app/inference.py::Inspector loads
# data/export/model.onnx and every uploaded image goes through the same canonical
# pipeline the station uses (resize -> bilateral -> normalise -> ONNX -> class
# thresholds -> connected regions -> geometry). Nothing here re-implements any of it.
#
# One deliberate difference from the station (docs/DEPLOYMENT.md §2): a public host
# means images travel over the internet to reach this container. The station's offline
# guarantee (NFR-05) is a property of where it runs, not of this code - the inference
# path still opens no outbound socket. Do not point a real production line at a public
# deployment; use the station install for that.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is onnxruntime's OpenMP runtime and is NOT in python:*-slim - without it the
# import fails at runtime with a bare "libgomp.so.1: cannot open shared object file".
# libglib2.0-0 is opencv's remaining shared dependency even in the headless build.
# Both are small and both are load-bearing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# requirements.txt already carries the inference core (onnxruntime,
# opencv-python-headless, scikit-image) alongside the web stack.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY data/metrics/coverage.json data/metrics/coverage.json
COPY data/export/model.onnx data/export/model.onnx

# The one file app/ needs from outside its own package. Inspector reads the input
# transform out of the graph's metadata (here: prep=bilateral) and loads it from
# bench/preprocess.py via `sys.path.insert(ROOT/"bench")` - see app/inference.py:109.
# That is deliberate: the app applies the *same* transform the weights were trained
# with rather than keeping a second copy that could drift (architecture §5.1). Without
# this file the graph loads and then fails with "No module named 'preprocess'".
COPY bench/preprocess.py bench/preprocess.py

# Fail the build loudly and early rather than at the first request. A missing or
# truncated model is the one input this image cannot substitute for, and the runtime
# symptom (Status reporting CHECK STATION, inspection stopped) is far less obvious
# than a build that stops here saying exactly what is wrong.
RUN python -c "import sys; from pathlib import Path; p = Path('data/export/model.onnx'); \
sys.exit('model.onnx is missing from the build context') if not p.exists() else None; \
sys.exit('model.onnx is truncated: %d bytes' % p.stat().st_size) if p.stat().st_size < 1000000 else None; \
print('model.onnx present: %.2f MB' % (p.stat().st_size / 1024 ** 2))"

# Two build-time steps, in this order.
#
# 1. Seed generates the demonstration history the History and Analytics screens read.
#    It runs the MOCK provider on purpose - those rows are procedurally generated
#    sample data, not measurements - and it happens at build time because seeding takes
#    minutes and a container should answer its first request immediately.
# 2. register_model then makes the real ONNX the active model_version, computing the
#    hash from the file itself. That is what lets MODEL_SHA256 stay empty below: the
#    database row becomes the reference, and status_service.model_check compares the
#    file on disk against it. A swapped model still stops inspection (T-23).
RUN INSPECTION_PROVIDER=mock DEMO_MODE=true python -m scripts.seed_db --live 60 \
 && python -m scripts.register_model \
      --model data/export/model.onnx \
      --version V12_22 \
      --params 1432000

# DEMO_MODE=true is safe alongside the real provider: live.html and status.html both
# gate the fake-station controls on `demo_mode and provider == "mock"`, so nothing
# renders a button that would 409. It marks this as a demonstration deployment while
# upload, camera capture and live inspection all run the real ONNX pipeline.
ENV INSPECTION_PROVIDER=real \
    DEMO_MODE=true \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    DATABASE_PATH=data/inspection.db \
    MODEL_PATH=data/export/model.onnx \
    MODEL_SHA256= \
    MIN_FREE_DISK_GB=0 \
    OMP_NUM_THREADS=1

EXPOSE 8000

# $PORT is what most container hosts (Render, Railway, Fly.io, Cloud Run) inject; 8000
# is the local fallback used everywhere else in this repository's docs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/healthz', timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
