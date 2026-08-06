"""Material coverage, thresholds and model metadata for the Materials screen.

Two rules from the Phase 2 report govern this module:

1. Coverage numbers are read from the evaluation metrics file (data/metrics/
   coverage.json), never typed into a template, so the screen cannot drift from the
   measured numbers (T-22).
2. Thresholds are read from app/postprocess.py::Profile at start-up and shown
   read-only.  They are not copied into the interface (T-06).
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.profiles import ACTIVE_PROFILE
from app.repositories import material_repository as materials
from app.repositories import model_repository as models

SUPPORT_LABELS = materials.SUPPORT_LABELS

#: Materials the interface must never present as evidence-backed.
LIMITED_STATUSES = {"one_product_only", "thin_coverage", "typing_unsupported", "not_supported"}


@lru_cache(maxsize=4)
def _load_metrics(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = settings.metrics_file
    if not path.exists():
        return {"materials": [], "model": {}, "overall": {}, "missing": True}
    return _load_metrics(str(path), path.stat().st_mtime)


def coverage(conn: sqlite3.Connection, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Join the measured coverage onto the material rows."""
    metrics = load_metrics(settings)
    by_code = {m["material_code"]: m for m in metrics.get("materials", [])}
    rows = []
    for row in materials.all_materials(conn):
        measured = by_code.get(row["material_code"], {})
        rows.append(
            {
                **row,
                "support_label": SUPPORT_LABELS.get(row["support_status"], row["support_status"]),
                "is_limited": row["support_status"] in LIMITED_STATUSES,
                "training_masks": measured.get("training_masks"),
                "training_masks_display": measured.get("training_masks_display", "\u2014"),
                "class_typing": measured.get("class_typing"),
                "cl_dice": measured.get("cl_dice"),
                "notes": measured.get("notes") or row.get("notes") or "",
            }
        )
    return rows


def thresholds(conn: sqlite3.Connection, material_code: str | None = None) -> dict[str, Any]:
    """The active threshold set.

    Stored profile values are returned so historical inspections resolve to the
    thresholds that produced them; the module values are returned alongside so the
    Materials screen can show that the two agree.
    """
    profile = materials.active_profile(conn, material_code) or {}
    module = ACTIVE_PROFILE.as_dict()
    stored = {
        "crack_threshold": profile.get("crack_threshold"),
        "scratch_threshold": profile.get("scratch_threshold"),
        "minimum_area_px": profile.get("minimum_area_px"),
        "minimum_skeleton_px": profile.get("minimum_skeleton_px"),
        "version_no": profile.get("version_no"),
        "material_code": profile.get("material_code"),
        "profile_id": profile.get("profile_id"),
    }
    in_sync = (
        stored["crack_threshold"] == module["crack_thresh"]
        and stored["scratch_threshold"] == module["scratch_thresh"]
        and stored["minimum_area_px"] == module["min_area_px"]
        and stored["minimum_skeleton_px"] == module["min_skeleton_px"]
    )
    return {
        "stored": stored,
        "module": module,
        "in_sync": in_sync,
        # The definition moved out of postprocess.py so the interface can read it
        # without importing cv2/numpy/scikit-image, which the mock deployment does not
        # install. postprocess.py re-exports the same object, so
        # `postprocess.Profile is profiles.Profile` — there is still one definition.
        "source": "app/profiles.py::Profile (re-exported by app/postprocess.py)",
    }


def model_info(conn: sqlite3.Connection, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    metrics = load_metrics(settings).get("model", {})
    row = models.active_model(conn) or {}
    sha = row.get("artefact_sha256") or metrics.get("artefact_sha256") or ""
    return {
        "file_name": row.get("file_name") or metrics.get("file_name"),
        "version": row.get("version") or metrics.get("version"),
        "artefact_sha256": sha,
        "sha_short": (sha[:4] + "\u2026" + sha[-3:]) if len(sha) > 10 else sha,
        "parameter_count": row.get("parameter_count") or metrics.get("parameter_count"),
        "size_mb": row.get("size_mb") or metrics.get("size_mb"),
        "precision": row.get("precision") or metrics.get("precision"),
        "latency_ms": row.get("latency_ms") or metrics.get("latency_ms"),
        "latency_conditions": metrics.get("latency_conditions"),
        "input_name": metrics.get("input_name"),
        "input_shape": metrics.get("input_shape"),
        "output_name": metrics.get("output_name"),
        "output_shape": metrics.get("output_shape"),
        "channel_order": metrics.get("channel_order"),
        "output_taxonomy": metrics.get("output_taxonomy", []),
        "activation": metrics.get("activation"),
        "arm_note": metrics.get("arm_note", "No ARM or phone latency measurement is currently claimed."),
        "file_present": settings.model_file.exists(),
        "configured_path": str(settings.model_file),
    }


def overall_metrics(settings: Settings | None = None) -> dict[str, Any]:
    return load_metrics(settings).get("overall", {})
