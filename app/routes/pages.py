"""Server-rendered pages - the six screens.

Every page renders complete on the server.  JavaScript refreshes the live result,
drives region navigation and polls batch progress, but no screen depends on it to show
its content: with scripting disabled the interface still reads.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import Settings
from app.dependencies import (
    NAV_ITEMS,
    PROVISIONAL_NOTE,
    get_db,
    inspection_view,
    page_numbers,
    settings_dep,
)
from app.repositories import batch_repository as batches
from app.repositories import inspection_repository as inspections
from app.services import (
    batch_service,
    history_service,
    inspection_service,
    material_service,
    status_service,
)
from app.services.history_service import STATUS_LABELS

router = APIRouter(tags=["pages"])


def _templates(request: Request):
    return request.app.state.templates


def _base_context(
    request: Request, conn: sqlite3.Connection, settings: Settings, active: str
) -> dict[str, Any]:
    return {
        "request": request,
        "nav_items": NAV_ITEMS,
        "active": active,
        "settings": settings,
        "station": settings.station_id,
        "health": status_service.health_strip(conn, settings),
        "provisional_note": PROVISIONAL_NOTE,
        "status_labels": STATUS_LABELS,
        "demo_mode": settings.demo_mode,
    }


@router.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/live", status_code=307)


@router.get("/live", response_class=HTMLResponse)
def live(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "live")
    row = inspections.latest(conn, settings.station_id) or inspections.latest(conn)
    context["view"] = inspection_view(conn, int(row["inspection_id"])) if row else None
    context["poll_ms"] = settings.live_poll_ms
    return _templates(request).TemplateResponse(request, "live.html", context)


@router.get("/capture", response_class=HTMLResponse)
def capture(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    """Camera capture and photo upload.

    The material list comes from the database rather than the template, so a material
    whose coverage is thin or unsupported is labelled as such at the point where the
    operator chooses it — not only afterwards on the Materials screen.
    """
    context = _base_context(request, conn, settings, "capture")
    context["materials"] = material_service.coverage(conn, settings)
    context["max_bytes"] = inspection_service.CAPTURE_MAX_BYTES
    return _templates(request).TemplateResponse(request, "capture.html", context)


@router.get("/regions", response_class=HTMLResponse)
def region_detail(
    request: Request,
    inspection_id: int | None = Query(None),
    region: int = Query(1, ge=1),
    mode: str = Query("overlay", pattern="^(overlay|source|side_by_side)$"),
    zoom: int = Query(3, ge=1, le=6),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "regions")

    if inspection_id is None:
        row = inspections.latest(conn, settings.station_id) or inspections.latest(conn)
        # Prefer something that actually has regions to look at.
        if row is None or int(row["region_count"] or 0) == 0:
            candidate = conn.execute(
                "SELECT inspection_id FROM inspection WHERE region_count > 0 "
                "ORDER BY captured_at DESC, inspection_id DESC LIMIT 1"
            ).fetchone()
            inspection_id = int(candidate["inspection_id"]) if candidate else (
                int(row["inspection_id"]) if row else None
            )
        else:
            inspection_id = int(row["inspection_id"])

    view = inspection_view(conn, inspection_id) if inspection_id else None
    context.update(
        {
            "view": view,
            "inspection_id": inspection_id,
            "mode": mode,
            "zoom": zoom,
            "selected_index": region,
        }
    )
    if view and view["regions"]:
        indices = [r["region_index"] for r in view["regions"]]
        selected = region if region in indices else indices[0]
        position = indices.index(selected)
        context.update(
            {
                "selected_index": selected,
                "selected_region": view["regions"][position],
                "prev_index": indices[position - 1] if position > 0 else None,
                "next_index": indices[position + 1] if position < len(indices) - 1 else None,
            }
        )
    return _templates(request).TemplateResponse(request, "region_detail.html", context)


@router.get("/batch", response_class=HTMLResponse)
def batch(
    request: Request,
    batch_run_id: int | None = Query(None),
    only_with_regions: bool = Query(False),
    status: str | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "batch")
    run_id = batch_run_id or batches.latest_id(conn)
    report = (
        batch_service.report(conn, run_id, status=status, only_with_regions=only_with_regions)
        if run_id
        else None
    )
    context.update(
        {
            "folders": batch_service.list_folders(settings),
            "materials": material_service.coverage(conn, settings),
            "runs": batches.recent(conn),
            "report": report,
            "batch_run_id": run_id,
            "only_with_regions": only_with_regions,
            "status_filter": status,
            "model": material_service.model_info(conn, settings),
        }
    )
    return _templates(request).TemplateResponse(request, "batch.html", context)


@router.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "history")
    filters = history_service.parse_filters(dict(request.query_params))
    page = history_service.search(conn, filters)
    context.update(
        {
            "filters": filters,
            "page": page,
            "options": history_service.options(conn),
            "totals": history_service.totals(conn, filters),
            "query": filters.query_string(),
            "page_numbers": page_numbers(page.page, page.page_count),
        }
    )
    return _templates(request).TemplateResponse(request, "history.html", context)


@router.get("/inspections/{inspection_id}", response_class=HTMLResponse)
def inspection_detail(
    request: Request,
    inspection_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    view = inspection_view(conn, inspection_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"No inspection {inspection_id}")
    context = _base_context(request, conn, settings, "history")
    context.update({"view": view, "back_query": request.query_params.get("back", "")})
    return _templates(request).TemplateResponse(request, "inspection_detail.html", context)


@router.get("/materials", response_class=HTMLResponse)
def materials(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "materials")
    context.update(
        {
            "materials": material_service.coverage(conn, settings),
            "thresholds": material_service.thresholds(conn),
            "model": material_service.model_info(conn, settings),
            "overall": material_service.overall_metrics(settings),
            "metrics_source": str(settings.metrics_file),
        }
    )
    return _templates(request).TemplateResponse(request, "materials.html", context)


@router.get("/status", response_class=HTMLResponse)
def status_page(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    context = _base_context(request, conn, settings, "status")
    context.update({"status": status_service.run_checks(conn, settings)})
    return _templates(request).TemplateResponse(request, "status.html", context)
