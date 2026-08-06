"""Schema creation, foreign keys, indexes and seed content."""

from __future__ import annotations

import sqlite3

import pytest

from app.database.connection import connect, foreign_keys_enabled
from app.database.migrations import EXPECTED_TABLES, init_database, verify_schema


def test_schema_creates_all_eight_tables(tmp_path):
    tables = init_database(tmp_path / "fresh.db")
    for name in EXPECTED_TABLES:
        assert name in tables
    assert len([t for t in tables if not t.startswith("sqlite_")]) == 8


def test_schema_creation_is_idempotent(tmp_path):
    path = tmp_path / "twice.db"
    first = init_database(path)
    second = init_database(path)
    assert first == second


def test_foreign_keys_are_enabled(conn):
    assert foreign_keys_enabled(conn)


def test_foreign_keys_are_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO defect_region (inspection_id, region_index, class_id, area_px, "
            "length_px, max_width_px, bbox_x, bbox_y, bbox_width, bbox_height) "
            "VALUES (999999, 1, 1, 10, 10, 2, 0, 0, 5, 5)"
        )
    conn.rollback()


def test_region_index_is_unique_per_inspection(conn):
    row = conn.execute(
        "SELECT inspection_id, region_index, class_id FROM defect_region LIMIT 1"
    ).fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO defect_region (inspection_id, region_index, class_id, area_px, "
            "length_px, max_width_px, bbox_x, bbox_y, bbox_width, bbox_height) "
            "VALUES (?, ?, ?, 10, 10, 2, 0, 0, 5, 5)",
            [row["inspection_id"], row["region_index"], row["class_id"]],
        )
    conn.rollback()


def test_status_column_rejects_an_unknown_state(conn):
    """A status outside the four allowed values must not be storable."""
    row = conn.execute("SELECT * FROM inspection LIMIT 1").fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO inspection (station_id, profile_id, model_version_id, captured_at, "
            "processed_at, status, region_count) VALUES (?, ?, ?, ?, ?, 'rejected', 0)",
            [row["station_id"], row["profile_id"], row["model_version_id"], "2026-01-01", "2026-01-01"],
        )
    conn.rollback()


def test_indexes_exist_for_history_and_batch_queries(conn):
    names = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    for expected in (
        "idx_inspection_captured_at",
        "idx_inspection_product_id",
        "idx_inspection_status",
        "idx_inspection_material",
        "idx_inspection_station",
        "idx_inspection_batch",
        "idx_region_inspection",
    ):
        assert expected in names


def test_verify_schema_reports_missing_tables(tmp_path):
    path = tmp_path / "partial.db"
    connection = connect(path)
    connection.execute("CREATE TABLE material (material_id INTEGER PRIMARY KEY)")
    missing = verify_schema(connection)
    connection.close()
    assert "inspection" in missing


def test_seed_creates_every_status_and_three_reference_sets(seeded, conn):
    statuses = {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) AS n FROM inspection GROUP BY status")
    }
    for status in ("regions_found", "clean", "acquisition_failure", "processing_failure"):
        assert statuses.get(status, 0) > 0, f"seed produced no {status} inspection"

    assert conn.execute("SELECT COUNT(*) AS n FROM material").fetchone()["n"] == 6
    assert conn.execute("SELECT COUNT(*) AS n FROM defect_class").fetchone()["n"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM station").fetchone()["n"] >= 3
    assert conn.execute("SELECT COUNT(*) AS n FROM batch_run").fetchone()["n"] >= 2


def test_seed_spans_multiple_dates_materials_and_stations(conn):
    dates = conn.execute("SELECT COUNT(DISTINCT substr(captured_at, 1, 10)) AS n FROM inspection").fetchone()["n"]
    materials = conn.execute("SELECT COUNT(DISTINCT material_id) AS n FROM inspection").fetchone()["n"]
    stations = conn.execute("SELECT COUNT(DISTINCT station_id) AS n FROM inspection").fetchone()["n"]
    assert dates >= 2
    assert materials >= 3
    assert stations >= 2


def test_region_count_matches_the_stored_regions(conn):
    """The denormalised counter must agree with the region table it summarises."""
    mismatches = conn.execute(
        """
        SELECT i.inspection_id FROM inspection i
        WHERE i.region_count <> (SELECT COUNT(*) FROM defect_region r WHERE r.inspection_id = i.inspection_id)
        """
    ).fetchall()
    assert mismatches == []


def test_failures_never_carry_regions(conn):
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM inspection "
        "WHERE status IN ('acquisition_failure', 'processing_failure') AND region_count > 0"
    ).fetchone()
    assert rows["n"] == 0


def test_clean_inspections_have_an_empty_region_list(conn):
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM inspection WHERE status = 'clean' AND region_count > 0"
    ).fetchone()
    assert rows["n"] == 0
