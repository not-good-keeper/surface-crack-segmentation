"""Status API: run the station checks and read the health strip."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import Settings
from app.dependencies import get_db, settings_dep
from app.services import status_service

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("")
def status(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    return status_service.run_checks(conn, settings)


@router.post("/check")
def run_checks(
    simulate: str | None = Query(None, description="demo-only fault injection; mock mode only"),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    if simulate is not None:
        if not settings.is_mock:
            raise HTTPException(
                status_code=403,
                detail="Fault simulation is a mock-mode demo control and is refused in real mode.",
            )
        if simulate not in status_service.SIMULATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown simulation '{simulate}'. Options: {', '.join(status_service.SIMULATIONS)}",
            )
        status_service.set_simulation(simulate)
    return status_service.run_checks(conn, settings)


@router.get("/strip")
def strip(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    return status_service.health_strip(conn, settings)
