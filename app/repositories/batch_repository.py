"""Batch runs.

The summary cards on the batch screen are computed from the inspection rows the run
produced, not from counters kept alongside them, so the cards and the export can never
disagree (FR-19 / T-16).  The counters on batch_run are written once at completion and
are only a convenience for the run list.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def create(conn: sqlite3.Connection, values: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO batch_run (source_folder, material_id, started_at, status, image_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            values["source_folder"],
            values.get("material_id"),
            values["started_at"],
            values.get("status", "running"),
            values.get("image_count", 0),
        ],
    )
    return int(cur.lastrowid)


def finish(conn: sqlite3.Connection, batch_run_id: int, values: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE batch_run
           SET finished_at = ?, image_count = ?, clean_count = ?,
               regions_found_count = ?, failure_count = ?, status = ?
         WHERE batch_run_id = ?
        """,
        [
            values.get("finished_at"),
            values.get("image_count", 0),
            values.get("clean_count", 0),
            values.get("regions_found_count", 0),
            values.get("failure_count", 0),
            values.get("status", "completed"),
            batch_run_id,
        ],
    )


def get(conn: sqlite3.Connection, batch_run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT b.*, m.material_code, m.material_name
        FROM batch_run b LEFT JOIN material m ON m.material_id = b.material_id
        WHERE b.batch_run_id = ?
        """,
        [batch_run_id],
    ).fetchone()
    return dict(row) if row else None


def recent(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT b.*, m.material_code, m.material_name
        FROM batch_run b LEFT JOIN material m ON m.material_id = b.material_id
        ORDER BY b.batch_run_id DESC LIMIT ?
        """,
        [limit],
    )
    return [dict(r) for r in rows]


def latest_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MAX(batch_run_id) AS id FROM batch_run").fetchone()
    return int(row["id"]) if row and row["id"] is not None else None
