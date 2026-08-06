"""Materials, defect classes and threshold profiles."""

from __future__ import annotations

import sqlite3
from typing import Any

SUPPORT_LABELS = {
    "supported": "supported",
    "one_product_only": "one product only",
    "thin_coverage": "thin coverage",
    "typing_unsupported": "typing unsupported",
    "not_supported": "not supported",
    "under_evaluation": "under evaluation",
}


def all_materials(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT material_id, material_code, material_name, support_status, notes "
        "FROM material ORDER BY material_id"
    )
    return [dict(r) for r in rows]


def by_code(conn: sqlite3.Connection, code: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT material_id, material_code, material_name, support_status, notes "
        "FROM material WHERE material_code = ?",
        [code],
    ).fetchone()
    return dict(row) if row else None


def material_id(conn: sqlite3.Connection, code: str | None) -> int | None:
    if not code:
        return None
    row = conn.execute("SELECT material_id FROM material WHERE material_code = ?", [code]).fetchone()
    return int(row["material_id"]) if row else None


def defect_classes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT class_id, class_code, display_name FROM defect_class ORDER BY class_id")
    return [dict(r) for r in rows]


def class_id(conn: sqlite3.Connection, code: str) -> int:
    row = conn.execute("SELECT class_id FROM defect_class WHERE class_code = ?", [code]).fetchone()
    if row is None:
        raise KeyError(f"Unknown defect class: {code}")
    return int(row["class_id"])


def active_profile(conn: sqlite3.Connection, material_code: str | None = None) -> dict[str, Any] | None:
    """The profile in force.

    A material-specific profile wins over the global one; both come from
    app/postprocess.py at seed time and are stored so an old inspection still resolves
    to the thresholds that produced it.
    """
    if material_code:
        row = conn.execute(
            """
            SELECT p.*, m.material_code FROM profile p
            LEFT JOIN material m ON m.material_id = p.material_id
            WHERE p.is_active = 1 AND m.material_code = ?
            ORDER BY p.version_no DESC LIMIT 1
            """,
            [material_code],
        ).fetchone()
        if row:
            return dict(row)
    row = conn.execute(
        """
        SELECT p.*, m.material_code FROM profile p
        LEFT JOIN material m ON m.material_id = p.material_id
        WHERE p.is_active = 1
        ORDER BY (p.material_id IS NULL) DESC, p.version_no DESC LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


def all_profiles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT p.*, m.material_code FROM profile p
        LEFT JOIN material m ON m.material_id = p.material_id
        ORDER BY p.profile_id
        """
    )
    return [dict(r) for r in rows]
