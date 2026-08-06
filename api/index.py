"""Vercel entry point.

Vercel's Python runtime looks for an ASGI application called ``app`` in this module.
Everything else is the same application that runs locally; nothing about the interface,
the database schema or the provider boundary changes for the deployment.

Read DEPLOYMENT.md before relying on this: a serverless deployment is a demonstration
of the interface, not a station deployment. The station runs on a CPU-only industrial
PC, offline, with a database that persists.
"""

import sys
from pathlib import Path

# The project root has to be importable: this file lives in api/, the package in app/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

__all__ = ["app"]
