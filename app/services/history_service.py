"""History queries - filtering, pagination and the filter option lists."""

from __future__ import annotations

import sqlite3
from typing import Any

from app.repositories import inspection_repository as inspections
from app.repositories.inspection_repository import HistoryFilters, Page

STATUS_LABELS = {
    "regions_found": "regions found",
    "clean": "clean",
    "acquisition_failure": "failed to read",
    "processing_failure": "processing failed",
}


def parse_filters(params: dict[str, Any]) -> HistoryFilters:
    """Build a filter set from query parameters, ignoring blanks.

    Filters live in the query string so a filtered view can be bookmarked, shared and
    returned to with the browser's back button.
    """

    def clean(name: str) -> str | None:
        value = params.get(name)
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def as_int(name: str, default: int) -> int:
        try:
            return int(str(params.get(name, default)).strip())
        except (TypeError, ValueError):
            return default

    status = clean("status")
    if status and status not in STATUS_LABELS:
        status = None
    return HistoryFilters(
        date_from=clean("date_from"),
        date_to=clean("date_to"),
        material=clean("material"),
        status=status,
        defect_class=clean("class"),
        station=clean("station"),
        product_id=clean("product_id"),
        page=max(1, as_int("page", 1)),
        page_size=max(5, min(200, as_int("page_size", 25))),
    )


def search(conn: sqlite3.Connection, filters: HistoryFilters) -> Page:
    return inspections.search(conn, filters)


def options(conn: sqlite3.Connection) -> dict[str, Any]:
    values = inspections.distinct_values(conn)
    values["statuses"] = [{"code": k, "label": v} for k, v in STATUS_LABELS.items()]
    return values


def totals(conn: sqlite3.Connection, filters: HistoryFilters) -> dict[str, int]:
    return inspections.status_totals(conn, filters)
