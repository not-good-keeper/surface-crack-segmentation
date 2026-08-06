"""Shared request dependencies and the view models the templates render.

Templates receive plain dictionaries assembled here rather than database rows, so a
schema change does not ripple into HTML, and so no template ever has to decide what a
status means.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from app.config import Settings, get_settings
from app.database.connection import connect
from app.services import inspection_service
from app.services.history_service import STATUS_LABELS

NAV_ITEMS = [
    {"key": "live", "label": "Live", "href": "/live", "description": "Result for the item at the station"},
    # Not in the Phase 2 wireframes: the six screens there assume a fixed camera above
    # a conveyor. Capture covers the case that has no station at all — a phone or a
    # laptop webcam pointed at a part, or a photograph from disk — which is how the
    # system is demonstrated and how the Phase 2 section 7 acceptance photographs were
    # taken. It is a separate screen so the Live wireframe stays exactly as drawn.
    {"key": "capture", "label": "Capture", "href": "/capture", "description": "Camera or photo, inspected now"},
    {"key": "regions", "label": "Regions", "href": "/regions", "description": "Region detail and measurements"},
    {"key": "batch", "label": "Batch", "href": "/batch", "description": "Run a folder and export the report"},
    {"key": "analytics", "label": "Analytics", "href": "/analytics", "description": "Charts and trends per session"},
    {"key": "history", "label": "History", "href": "/history", "description": "Search past inspections"},
    {"key": "materials", "label": "Materials", "href": "/materials", "description": "Coverage, thresholds, model"},
    {"key": "status", "label": "Status", "href": "/status", "description": "Camera, model, database, disk"},
]

#: Shown wherever a class label appears.  The mask is the primary output; typing is
#: about 49 % correct on crack pixels, so the label is marked provisional everywhere.
PROVISIONAL_NOTE = "Class label is provisional \u2014 see Materials for measured coverage by material."


def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def settings_dep() -> Settings:
    return get_settings()


def status_label(status: str | None) -> str:
    return STATUS_LABELS.get(status or "", status or "\u2014")


def px(value: Any, digits: int = 0) -> str:
    if value is None:
        return "\u2014"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "\u2014"
    return f"{number:,.{digits}f} px"


def number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def short_time(value: str | None) -> str:
    """'12 Aug 11:42:07' - the history table format from the Phase 2 wireframe."""
    if not value:
        return "\u2014"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(value)
        return dt.strftime("%d %b %H:%M:%S")
    except ValueError:
        return value


def split_last(value: Any, separator: str = "/") -> str:
    """Last path segment - filenames in tables, without leaking the server's layout."""
    if not value:
        return "\u2014"
    return str(value).replace("\\", "/").rstrip("/").split(separator)[-1]


def page_numbers(current: int, total: int, window: int = 2) -> list[int]:
    """Page links with 0 used as an ellipsis marker."""
    if total <= 1:
        return [1] if total == 1 else []
    pages = {1, total}
    pages.update(range(max(1, current - window), min(total, current + window) + 1))
    ordered = sorted(pages)
    out: list[int] = []
    previous = 0
    for number in ordered:
        if previous and number - previous > 1:
            out.append(0)
        out.append(number)
        previous = number
    return out


def class_breakdown_text(breakdown: dict[str, int]) -> str:
    parts = [f"{n} {code}" for code, n in sorted(breakdown.items()) if n]
    return ", ".join(parts) if parts else "\u2014"


def inspection_view(conn: sqlite3.Connection, inspection_id: int) -> dict[str, Any] | None:
    """Assemble the full view model for one inspection."""
    payload = inspection_service.inspection_payload(conn, inspection_id)
    if payload is None:
        return None
    row = payload["inspection"]
    payload["has_source"] = bool(row.get("source_image_path"))
    payload["has_overlay"] = bool(row.get("overlay_image_path"))
    payload["source_url"] = f"/media/inspection/{inspection_id}/source"
    payload["overlay_url"] = f"/media/inspection/{inspection_id}/overlay"
    payload["status_label"] = status_label(row["status"])
    payload["breakdown_text"] = class_breakdown_text(payload["class_breakdown"])
    return payload


def api_inspection(payload: dict[str, Any]) -> dict[str, Any]:
    """The JSON shape returned by /api/live and /api/inspections/{id}."""
    row = payload["inspection"]
    return {
        "inspection_id": row["inspection_id"],
        "product_id": row["product_id"],
        "material": row["material_code"],
        "material_support_status": row["support_status"],
        "station": row["station_code"],
        "captured_at": row["captured_at"],
        "processed_at": row["processed_at"],
        "status": row["status"],
        "status_label": status_label(row["status"]),
        "region_count": row["region_count"],
        "latency_ms": row["latency_ms"],
        "image_sha256": row["image_sha256"],
        "error_code": row["error_code"],
        "error_message": row["error_message"],
        "batch_run_id": row["batch_run_id"],
        "max_length_px": row["max_length_px"],
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
        "summary": payload["summary"],
        "class_breakdown": payload["class_breakdown"],
        "regions": [
            {
                "region_index": r["region_index"],
                "class_code": r["class_code"],
                "display_name": r["display_name"],
                "area_px": r["area_px"],
                "length_px": r["length_px"],
                "max_width_px": r["max_width_px"],
                "bbox": {
                    "x": r["bbox_x"],
                    "y": r["bbox_y"],
                    "width": r["bbox_width"],
                    "height": r["bbox_height"],
                },
                "centroid": {"x": r["centroid_x"], "y": r["centroid_y"]},
            }
            for r in payload["regions"]
        ],
        "source_image_url": payload["source_url"] if payload["has_source"] else None,
        "overlay_image_url": payload["overlay_url"] if payload["has_overlay"] else None,
        "provisional_note": PROVISIONAL_NOTE,
    }
