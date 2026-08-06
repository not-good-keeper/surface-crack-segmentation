# Container image for a hosted, public-facing demo of the interface.
#
# Mock mode only. The real inspection pipeline stays off any public host by design
# (NFR-05): factory images must never leave the site, and app/inference.py opens no
# socket by construction. This image runs INSPECTION_PROVIDER=mock and demonstration
# data generated procedurally at build time by app/providers/mock_assets.py - nothing
# downloaded, nothing real. See docs/DEPLOYMENT.md for what a hosted demo does and
# does not show, and LOCAL_SETUP.md / README.md for the real station install.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Mock mode never imports onnxruntime, opencv-python-headless or scikit-image - see
# requirements-real.txt. Installing only what mock mode needs keeps this image a few
# tens of MB instead of the ~290 MB the real inference core adds, for a stack a public
# host must never run anyway.
RUN pip install \
    "fastapi>=0.111,<1.0" \
    "uvicorn[standard]>=0.30" \
    "jinja2>=3.1,<4.0" \
    "pydantic>=2.7,<3.0" \
    "pydantic-settings>=2.2,<3.0" \
    "python-multipart>=0.0.9" \
    "numpy>=1.26" \
    "pillow>=10.3"

COPY app/ app/
COPY scripts/ scripts/
COPY data/metrics/coverage.json data/metrics/coverage.json

ENV INSPECTION_PROVIDER=mock \
    DEMO_MODE=true \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    DATABASE_PATH=data/inspection.db \
    MIN_FREE_DISK_GB=0

# Seed once at build time rather than on first request: the mock seed takes about two
# minutes, and a container should answer its first request immediately. This is the
# one part of the old Vercel bundle approach worth keeping - build_demo_bundle.py did
# the same thing because a serverless cold start had seconds, not minutes - except a
# Docker build has an actual build step, so nothing needs to be committed to git for it.
RUN python -m scripts.seed_db

EXPOSE 8000

# $PORT is what most container hosts (Render, Railway, Fly.io's fly.toml, Cloud Run)
# inject; 8000 is the local fallback used everywhere else in this repository's docs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/healthz', timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
