"""Model versions."""

from __future__ import annotations

import sqlite3
from typing import Any


def active_model(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM model_version WHERE is_active = 1 ORDER BY model_version_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM model_version ORDER BY model_version_id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def all_models(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM model_version ORDER BY model_version_id")]


def by_sha(conn: sqlite3.Connection, sha256: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM model_version WHERE artefact_sha256 = ?", [sha256]).fetchone()
    return dict(row) if row else None


def stations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM station ORDER BY station_id")]


def station_by_code(conn: sqlite3.Connection, code: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM station WHERE station_code = ?", [code]).fetchone()
    return dict(row) if row else None
