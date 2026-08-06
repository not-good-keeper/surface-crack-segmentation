"""Materials, thresholds and model metadata API.

Every number here originates in the evaluation metrics file or in
app/postprocess.py::Profile.  Nothing is typed into a template or a script.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.config import Settings
from app.dependencies import get_db, settings_dep
from app.services import material_service

router = APIRouter(prefix="/api", tags=["materials"])


@router.get("/materials")
def materials(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    return {
        "materials": material_service.coverage(conn, settings),
        "thresholds": material_service.thresholds(conn),
        "overall": material_service.overall_metrics(settings),
        "metrics_source": str(settings.metrics_file),
    }


@router.get("/thresholds")
def thresholds(
    material: str | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Threshold values, read from the module at start-up and stored per profile.

    The frontend calls this rather than holding its own copy, so re-tuning a threshold
    in app/postprocess.py changes the interface with no other edit (T-06).
    """
    return material_service.thresholds(conn, material)


@router.get("/model")
def model(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    return material_service.model_info(conn, settings)
