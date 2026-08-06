"""Inspection API: live result, demo advance, history query, detail, region detail."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from app.config import Settings
from app.dependencies import api_inspection, get_db, inspection_view, settings_dep
from app.repositories import inspection_repository as inspections
from app.services import export_service, history_service, inspection_service, status_service

router = APIRouter(prefix="/api", tags=["inspections"])


@router.get("/live")
def live(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """The result for the item currently at the station.

    When a station check is failing the payload says so and the screen shows
    CHECK STATION instead of an inspection result - a failing check is never presented
    as a normal state.
    """
    health = status_service.health_strip(conn, settings)
    row = inspections.latest(conn, settings.station_id) or inspections.latest(conn)

    payload: dict[str, Any] = {
        "station": settings.station_id,
        "health": health,
        "station_ok": not health["blocks_inspection"],
        "demo_mode": settings.demo_mode,
        "provider": settings.inspection_provider,
        "poll_ms": settings.live_poll_ms,
        "inspection": None,
    }
    if row is not None:
        view = inspection_view(conn, int(row["inspection_id"]))
        if view:
            payload["inspection"] = api_inspection(view)
    return payload


@router.post("/demo/next")
def demo_next(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
    force: str | None = Query(None, description="force a scenario: clean, crack_only, scratch_only, mixed, acquisition_failure, processing_failure"),
) -> dict[str, Any]:
    """Advance the mock station by one part.

    Only available in demo mode.  This is a developer control, not an operator action:
    a real station advances when the line advances.
    """
    if not settings.demo_mode:
        raise HTTPException(status_code=403, detail="DEMO_MODE is off; the demo control is disabled.")
    if not settings.is_mock:
        raise HTTPException(
            status_code=409,
            detail="The demo control only advances the mock provider. In real mode the line advances the station.",
        )
    health = status_service.health_strip(conn, settings)
    if health["blocks_inspection"]:
        raise HTTPException(
            status_code=409,
            detail=f"Inspection is stopped: {', '.join(health['failing'])}. Clear the failing check first.",
        )

    inspection_id = inspection_service.next_demo_inspection(conn, settings, force_kind=force)
    conn.commit()
    view = inspection_view(conn, inspection_id)
    return {"inspection_id": inspection_id, "inspection": api_inspection(view) if view else None}


@router.post("/inspections/capture")
async def capture(
    file: UploadFile = File(..., description="One frame: a camera capture or an uploaded photo"),
    material: str = Form("steel"),
    product_id: str | None = Form(None),
    source: str = Form("upload"),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Inspect one frame submitted from the browser.

    The device camera and the file picker both arrive here; they differ only in where
    the browser obtained the bytes, so `source` is recorded and nothing else branches
    on it. The frame is handed to the configured provider and stored exactly like a
    station inspection, which is why a capture shows up in History and in the exports.

    A station check that blocks inspection blocks this too. Producing a result while
    the model hash does not match would be producing it from a file the system cannot
    identify.
    """
    health = status_service.health_strip(conn, settings)
    if health["blocks_inspection"]:
        raise HTTPException(
            status_code=409,
            detail=f"Inspection is stopped: {', '.join(health['failing'])}. Clear the failing check first.",
        )

    if source not in ("camera", "upload"):
        raise HTTPException(status_code=422, detail="source must be 'camera' or 'upload'")

    image_bytes = await file.read()
    try:
        inspection_id = inspection_service.capture_inspection(
            conn,
            settings,
            image_bytes=image_bytes,
            filename=file.filename,
            material=material,
            product_id=product_id,
            source=source,
        )
    except inspection_service.CaptureTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    conn.commit()
    view = inspection_view(conn, inspection_id)
    return {
        "inspection_id": inspection_id,
        "inspection": api_inspection(view) if view else None,
        "provider": settings.inspection_provider,
        # The interface renders this as an unmissable notice. A generated region drawn
        # on a photograph the operator just took is the single easiest thing in this
        # application to mistake for a measurement.
        "generated": settings.is_mock,
        "region_detail_url": f"/regions?inspection_id={inspection_id}&region=1",
    }


@router.get("/inspections")
def list_inspections(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    filters = history_service.parse_filters(dict(request.query_params))
    page = history_service.search(conn, filters)
    return {
        "total": page.total,
        "page": page.page,
        "page_size": page.page_size,
        "page_count": page.page_count,
        "filters": filters.active(),
        "totals": history_service.totals(conn, filters),
        "items": [
            {
                "inspection_id": r["inspection_id"],
                "captured_at": r["captured_at"],
                "product_id": r["product_id"],
                "material": r["material_code"],
                "station": r["station_code"],
                "status": r["status"],
                "region_count": r["region_count"],
                "max_length_px": r["max_length_px"],
                "model_version": r["model_version"],
                "error_code": r["error_code"],
            }
            for r in page.rows
        ],
    }


@router.get("/inspections/export.csv")
def export_history_csv(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    """CSV for the current filter set - the same query the table uses."""
    filters = history_service.parse_filters(dict(request.query_params))
    body = export_service.to_csv(conn, filters)
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="inspection-history.csv"'},
    )


@router.get("/inspections/export.json")
def export_history_json(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    filters = history_service.parse_filters(dict(request.query_params))
    body = export_service.to_json(conn, filters, context={"export": "inspection-history"})
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="inspection-history.json"'},
    )


@router.get("/inspections/{inspection_id}")
def get_inspection(
    inspection_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    view = inspection_view(conn, inspection_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"No inspection {inspection_id}")
    return api_inspection(view)


@router.get("/inspections/{inspection_id}/regions/{region_index}")
def get_region(
    inspection_id: int,
    region_index: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    view = inspection_view(conn, inspection_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"No inspection {inspection_id}")
    regions = view["regions"]
    match = next((r for r in regions if r["region_index"] == region_index), None)
    if match is None:
        raise HTTPException(
            status_code=404, detail=f"Inspection {inspection_id} has no region {region_index}"
        )
    indices = [r["region_index"] for r in regions]
    position = indices.index(region_index)
    row = view["inspection"]
    return {
        "inspection_id": inspection_id,
        "region_count": len(regions),
        "prev_index": indices[position - 1] if position > 0 else None,
        "next_index": indices[position + 1] if position < len(indices) - 1 else None,
        "region": {
            "region_index": match["region_index"],
            "class_code": match["class_code"],
            "display_name": match["display_name"],
            "area_px": match["area_px"],
            "length_px": match["length_px"],
            "max_width_px": match["max_width_px"],
            "bbox": {
                "x": match["bbox_x"],
                "y": match["bbox_y"],
                "width": match["bbox_width"],
                "height": match["bbox_height"],
            },
            "centroid": {"x": match["centroid_x"], "y": match["centroid_y"]},
        },
        "product_id": row["product_id"],
        "station": row["station_code"],
        "material": row["material_code"],
        "captured_at": row["captured_at"],
        "image_sha256": row["image_sha256"],
        "model": {
            "version": row["model_version"],
            "file_name": row["model_file_name"],
            "artefact_sha256": row["artefact_sha256"],
        },
        "profile": {
            "version_no": row["profile_version"],
            "crack_threshold": row["crack_threshold"],
            "scratch_threshold": row["scratch_threshold"],
            "minimum_area_px": row["minimum_area_px"],
            "minimum_skeleton_px": row["minimum_skeleton_px"],
        },
        "crop_url": f"/media/inspection/{inspection_id}/region/{region_index}",
        "measurement_note": (
            "Maximum width is the widest inscribable point measured along the region "
            "skeleton. It may differ from the widest visual extent."
        ),
    }
