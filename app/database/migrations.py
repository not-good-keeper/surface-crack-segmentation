"""Database initialisation.

The schema is created with ``CREATE TABLE IF NOT EXISTS`` so applying it is idempotent
and safe on every start-up.  There is no versioned migration chain yet because nothing
has shipped; ``DATABASE.md`` describes the procedure for the first real migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.database.connection import connect, create_schema, table_names

EXPECTED_TABLES = [
    "batch_run",
    "defect_class",
    "defect_region",
    "inspection",
    "material",
    "model_version",
    "profile",
    "station",
]


def init_database(db_path: Path | str | None = None) -> list[str]:
    """Create the schema if needed and return the resulting table list."""
    conn = connect(db_path)
    try:
        create_schema(conn)
        return table_names(conn)
    finally:
        conn.close()


def verify_schema(conn: sqlite3.Connection) -> list[str]:
    """Return the list of expected tables that are missing."""
    present = set(table_names(conn))
    return [t for t in EXPECTED_TABLES if t not in present]


def database_is_seeded(conn: sqlite3.Connection) -> bool:
    try:
        return conn.execute("SELECT COUNT(*) FROM inspection").fetchone()[0] > 0
    except sqlite3.Error:
        return False
