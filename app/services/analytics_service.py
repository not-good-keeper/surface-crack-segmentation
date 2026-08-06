"""Session (batch run) analytics: a cross-session overview and one dashboard per run.

"Session" here means the batch_run - a start-to-finish processing job over a folder,
the one place the schema already groups many inspections under a single id (schema.sql
§6/§7, and FR-19's reconciliation rule). Live and Capture inspections exist too, but
nothing groups them into a session, so this module aggregates over batch_run rows only.

Every number here is computed from the same inspection/defect_region rows the Batch
report and its exports use (batch_service.compute_totals), so a chart on this page and
the totals on /batch can never disagree.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from app.repositories import batch_repository as batches
from app.services import batch_service, charts

STATUS_COLORS = {
    "clean": "var(--ok)",
    "regions_found": "var(--warn)",
    "acquisition_failure": "var(--alert)",
    "processing_failure": "var(--ink-2)",
}
STATUS_LABELS_SHORT = {
    "clean": "Clean",
    "regions_found": "Regions found",
    "acquisition_failure": "Acquisition failure",
    "processing_failure": "Processing failure",
}
STATUS_ORDER = ("clean", "regions_found", "acquisition_failure", "processing_failure")
CRACK_COLOR = "var(--crack)"
SCRATCH_COLOR = "var(--scratch)"


def _status_segments(totals: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": STATUS_LABELS_SHORT[key], "value": totals.get(key, 0) or 0, "color": STATUS_COLORS[key]}
        for key in STATUS_ORDER
    ]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def latency_stats(conn: sqlite3.Connection, batch_run_id: int) -> dict[str, float | int | None]:
    rows = conn.execute(
        "SELECT latency_ms FROM inspection WHERE batch_run_id = ? AND latency_ms IS NOT NULL ORDER BY latency_ms",
        [batch_run_id],
    ).fetchall()
    values = [float(r["latency_ms"]) for r in rows]
    if not values:
        return {"avg": None, "median": None, "p95": None, "max": None, "count": 0}
    return {
        "avg": round(sum(values) / len(values), 1),
        "median": round(_percentile(values, 0.5), 1),
        "p95": round(_percentile(values, 0.95), 1),
        "max": round(max(values), 1),
        "count": len(values),
    }


def _timeline_points(conn: sqlite3.Connection, batch_run_id: int, started_at: str | None) -> list[tuple[float, int]]:
    """Cumulative processed count over elapsed run minutes, for the run's trend line."""
    rows = conn.execute(
        "SELECT processed_at FROM inspection WHERE batch_run_id = ? ORDER BY processed_at, inspection_id",
        [batch_run_id],
    ).fetchall()
    start = _parse_dt(started_at)
    points: list[tuple[float, int]] = []
    for i, row in enumerate(rows, start=1):
        at = _parse_dt(row["processed_at"])
        elapsed_min = max(0.0, (at - start).total_seconds() / 60) if (at and start) else float(i - 1)
        points.append((elapsed_min, i))
    return points


def session_dashboard(conn: sqlite3.Connection, batch_run_id: int) -> dict[str, Any] | None:
    """Everything the per-session analytics page needs, charts included."""
    run = batches.get(conn, batch_run_id)
    if run is None:
        return None

    totals = batch_service.compute_totals(conn, batch_run_id)
    points = _timeline_points(conn, batch_run_id, run.get("started_at"))
    latency = latency_stats(conn, batch_run_id)

    duration_min = None
    throughput = None
    start = _parse_dt(run.get("started_at"))
    end = _parse_dt(run.get("finished_at"))
    if start and end and end > start:
        duration_min = (end - start).total_seconds() / 60
        if duration_min and totals["processed"]:
            throughput = round(totals["processed"] / duration_min, 1)

    class_bars = [
        {"label": "Crack", "value": totals["crack_regions"], "color": CRACK_COLOR},
        {"label": "Scratch", "value": totals["scratch_regions"], "color": SCRATCH_COLOR},
    ]

    charts_svg = {
        "status_mix": charts.stacked_bar(_status_segments(totals), title=f"Outcome mix for run {batch_run_id}"),
        "class_breakdown": charts.bar_chart(class_bars),
        "timeline": (
            charts.line_area(
                [p[1] for p in points],
                [f"{p[0]:.0f} min" for p in points],
                unit=" images",
            )
            if len(points) >= 2
            else charts.empty_state("Not enough data yet for a trend")
        ),
    }

    return {
        "run": run,
        "totals": totals,
        "duration_min": round(duration_min, 1) if duration_min is not None else None,
        "throughput": throughput,
        "latency": latency,
        "charts": charts_svg,
        "status_legend": _status_segments(totals),
        "class_legend": [
            {"label": "Crack", "color": CRACK_COLOR},
            {"label": "Scratch", "color": SCRATCH_COLOR},
        ],
    }


def overview(conn: sqlite3.Connection, limit: int = 15) -> dict[str, Any]:
    """The cross-session landing page: recent sessions plus deployment-wide totals."""
    recent = batches.recent(conn, limit)
    sessions = [{"run": run, "totals": batch_service.compute_totals(conn, run["batch_run_id"])} for run in recent]
    sessions.sort(key=lambda s: s["run"]["batch_run_id"])  # chronological for the trend charts

    agg_rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM inspection WHERE batch_run_id IS NOT NULL GROUP BY status"
    ).fetchall()
    agg_totals = {r["status"]: int(r["n"]) for r in agg_rows}
    for key in STATUS_ORDER:
        agg_totals.setdefault(key, 0)
    agg_processed = sum(agg_totals[k] for k in STATUS_ORDER)

    region_rows = conn.execute(
        """
        SELECT c.class_code AS class_code, COUNT(*) AS n
        FROM defect_region r
        JOIN defect_class c ON c.class_id = r.class_id
        JOIN inspection i ON i.inspection_id = r.inspection_id
        WHERE i.batch_run_id IS NOT NULL
        GROUP BY c.class_code
        """
    ).fetchall()
    agg_regions = {r["class_code"]: int(r["n"]) for r in region_rows}
    agg_regions.setdefault("crack", 0)
    agg_regions.setdefault("scratch", 0)

    session_count = int(conn.execute("SELECT COUNT(*) AS n FROM batch_run").fetchone()["n"])

    clean_pct_trend = [
        round(s["totals"]["clean"] / s["totals"]["processed"] * 100, 1) if s["totals"]["processed"] else 0.0
        for s in sessions
    ]
    trend_labels = [f"Run {s['run']['batch_run_id']}" for s in sessions]
    categories = [str(s["run"]["batch_run_id"]) for s in sessions]

    charts_svg = {
        "status_mix": charts.stacked_bar(
            [{"label": STATUS_LABELS_SHORT[k], "value": agg_totals[k], "color": STATUS_COLORS[k]} for k in STATUS_ORDER],
            title="Outcome mix across every session",
        ),
        "clean_trend": (
            charts.line_area(clean_pct_trend, trend_labels, color="var(--ok)", unit="%", y_min=0, y_max=100)
            if len(clean_pct_trend) >= 2
            else charts.empty_state("Run at least two sessions to see a trend")
        ),
        "defect_mix": (
            charts.grouped_bar_chart(
                categories,
                [
                    {"name": "Crack", "color": CRACK_COLOR, "values": [s["totals"]["crack_regions"] for s in sessions]},
                    {"name": "Scratch", "color": SCRATCH_COLOR, "values": [s["totals"]["scratch_regions"] for s in sessions]},
                ],
            )
            if categories
            else charts.empty_state("No sessions yet")
        ),
    }

    return {
        "sessions": list(reversed(sessions)),  # most recent first for the table
        "session_count": session_count,
        "agg_totals": agg_totals,
        "agg_processed": agg_processed,
        "agg_regions": agg_regions,
        "agg_clean_pct": round(agg_totals["clean"] / agg_processed * 100, 1) if agg_processed else 0.0,
        "agg_regions_pct": round(agg_totals["regions_found"] / agg_processed * 100, 1) if agg_processed else 0.0,
        "charts": charts_svg,
        "status_legend": [{"label": STATUS_LABELS_SHORT[k], "color": STATUS_COLORS[k]} for k in STATUS_ORDER],
        "class_legend": [
            {"label": "Crack", "color": CRACK_COLOR},
            {"label": "Scratch", "color": SCRATCH_COLOR},
        ],
    }
