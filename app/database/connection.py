"""SQLite connection handling.

Plain ``sqlite3`` and plain SQL, no ORM (Phase 2 report, section 8.3: a two-person
team maintains this).  Foreign keys are enabled on every connection - SQLite disables
them by default per connection, not per database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows returned as mappings."""
    settings = get_settings()
    path = Path(db_path) if db_path else settings.db_file
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.commit()


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def foreign_keys_enabled(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
