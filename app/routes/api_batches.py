"""Batch API: start a run, poll it, read the report, export it."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.config import Settings
from app.dependencies import get_db, settings_dep
from app.repositories import batch_repository as batches
from app.schemas.batch import BatchStartIn
from app.services import batch_service, export_service

router = APIRouter(prefix="/api/batches", tags=["batches"])


@router.get("")
def list_batches(conn: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    return {
        "runs": batches.recent(conn),
        "folders": batch_service.list_folders(),
    }


@router.post("")
def start_batch(
    payload: BatchStartIn,
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Start a run over a folder inside BATCH_ROOT.

    A folder outside the configured root is refused with 400 rather than followed.
    """
    try:
        info = batch_service.start_run(
            payload.source_folder,
            payload.material,
            product_prefix=payload.product_prefix,
            dry_run=payload.dry_run,
            settings=settings,
            # A serverless host freezes background threads once the response is sent,
            # so the work has to finish inside the request there.
            synchronous=settings.run_batches_synchronously,
        )
    except batch_service.UnsafeFolder as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return info


@router.get("/{batch_run_id}")
def get_batch(
    batch_run_id: int,
    only_with_regions: bool = Query(False),
    status: str | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    report = batch_service.report(
        conn, batch_run_id, status=status, only_with_regions=only_with_regions
    )
    if report is None:
        raise HTTPException(status_code=404, detail=f"No batch run {batch_run_id}")
    return {
        "batch_run_id": batch_run_id,
        "run": report["run"],
        "totals": report["totals"],
        "filtered_totals": report["filtered_totals"],
        "progress": report["progress"],
        "filter": report["filter"],
        "rows": [
            {
                "inspection_id": r["inspection_id"],
                "image": (r["source_image_path"] or "").split("/")[-1] or "\u2014",
                "product_id": r["product_id"],
                "material": r["material_code"],
                "status": r["status"],
                "region_count": r["region_count"],
                "class_breakdown": r["class_breakdown"],
                "max_length_px": r["max_length_px"],
                "latency_ms": r["latency_ms"],
                "error_code": r["error_code"],
            }
            for r in report["rows"]
        ],
    }


@router.get("/{batch_run_id}/export.csv")
def export_csv(
    batch_run_id: int,
    only_with_regions: bool = Query(False),
    status: str | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    if batches.get(conn, batch_run_id) is None:
        raise HTTPException(status_code=404, detail=f"No batch run {batch_run_id}")
    filters = batch_service.batch_filters(
        batch_run_id, status="regions_found" if only_with_regions else status
    )
    body = export_service.to_csv(conn, filters)
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_run_id}.csv"'},
    )


@router.get("/{batch_run_id}/export.json")
def export_json(
    batch_run_id: int,
    only_with_regions: bool = Query(False),
    status: str | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    run = batches.get(conn, batch_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No batch run {batch_run_id}")
    filters = batch_service.batch_filters(
        batch_run_id, status="regions_found" if only_with_regions else status
    )
    body = export_service.to_json(
        conn, filters, context={"batch_run_id": batch_run_id, "source_folder": run["source_folder"]}
    )
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_run_id}.json"'},
    )
