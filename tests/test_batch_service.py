"""Batch runs: totals reconcile, statuses stay separate, folders stay inside the root."""

from __future__ import annotations

import pytest

from app.repositories import batch_repository as batches
from app.services import batch_service


def latest_run(conn):
    return batches.recent(conn, limit=1)[0]


def test_seed_created_batch_runs(conn):
    assert len(batches.recent(conn)) >= 2


def test_totals_reconcile_with_the_stored_rows(conn):
    run = latest_run(conn)
    run_id = run["batch_run_id"]
    totals = batch_service.compute_totals(conn, run_id)

    counted = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM inspection WHERE batch_run_id = ? GROUP BY status",
            [run_id],
        )
    }
    assert totals["processed"] == sum(counted.values())
    assert totals["clean"] == counted.get("clean", 0)
    assert totals["regions_found"] == counted.get("regions_found", 0)
    assert totals["failed"] == counted.get("acquisition_failure", 0) + counted.get("processing_failure", 0)


def test_processed_equals_the_sum_of_the_four_statuses(conn):
    for run in batches.recent(conn):
        totals = batch_service.compute_totals(conn, run["batch_run_id"])
        assert totals["processed"] == (
            totals["regions_found"] + totals["clean"]
            + totals["acquisition_failure"] + totals["processing_failure"]
        )


def test_a_failed_file_is_never_counted_as_clean(conn):
    run = latest_run(conn)
    rows = batch_service.results(conn, run["batch_run_id"])
    failures = [r for r in rows if r["status"] in ("acquisition_failure", "processing_failure")]
    assert failures, "the seeded batch folders contain genuinely unreadable files"
    for row in failures:
        assert row["status"] != "clean"
        assert row["region_count"] == 0
        assert row["error_code"]

    totals = batch_service.compute_totals(conn, run["batch_run_id"])
    assert totals["clean"] + totals["failed"] <= totals["processed"]


def test_region_totals_match_the_region_rows(conn):
    run = latest_run(conn)
    totals = batch_service.compute_totals(conn, run["batch_run_id"])
    counted = conn.execute(
        "SELECT COUNT(*) AS n FROM defect_region r "
        "WHERE r.inspection_id IN (SELECT inspection_id FROM inspection WHERE batch_run_id = ?)",
        [run["batch_run_id"]],
    ).fetchone()["n"]
    assert totals["regions_total"] == counted
    assert totals["crack_regions"] + totals["scratch_regions"] == counted


def test_only_with_regions_filter(conn):
    run = latest_run(conn)
    rows = batch_service.results(conn, run["batch_run_id"], only_with_regions=True)
    assert all(r["status"] == "regions_found" and r["region_count"] > 0 for r in rows)


def test_report_includes_run_totals_and_rows(conn):
    run = latest_run(conn)
    report = batch_service.report(conn, run["batch_run_id"])
    assert report["run"]["batch_run_id"] == run["batch_run_id"]
    assert report["totals"]["processed"] == len(report["rows"])


def test_report_for_an_unknown_run_is_none(conn):
    assert batch_service.report(conn, 999999) is None


@pytest.mark.parametrize(
    "folder",
    ["../../etc", "/etc", "..", "steel/../../../..", "/etc/passwd"],
)
def test_folders_outside_the_batch_root_are_refused(settings, folder):
    with pytest.raises(batch_service.UnsafeFolder):
        batch_service.resolve_folder(folder, settings)


def test_a_missing_folder_is_refused(settings):
    with pytest.raises(batch_service.UnsafeFolder):
        batch_service.resolve_folder("no-such-folder", settings)


def test_a_configured_folder_resolves(settings):
    folders = batch_service.list_folders(settings)
    resolved = batch_service.resolve_folder(folders[0]["relative"], settings)
    assert resolved.is_dir()
    assert resolved.is_relative_to(settings.batch_dir.resolve())


def test_non_image_extensions_are_skipped(settings, temp_batch_folder):
    (temp_batch_folder / "notes.txt").write_text("not an image")
    (temp_batch_folder / "archive.zip").write_bytes(b"PK\x03\x04")
    assert batch_service.image_files(temp_batch_folder) == []


def test_dry_run_counts_files_without_writing_inspections(settings, temp_batch_folder, conn):
    from PIL import Image

    for index in range(3):
        Image.new("RGB", (64, 64), (100, 100, 100)).save(temp_batch_folder / f"img{index}.png")

    before = conn.execute("SELECT COUNT(*) AS n FROM inspection").fetchone()["n"]
    info = batch_service.start_run(
        temp_batch_folder.name, "steel", dry_run=True, settings=settings, synchronous=True
    )
    after = conn.execute("SELECT COUNT(*) AS n FROM inspection").fetchone()["n"]

    assert info["dry_run"] is True
    assert info["image_count"] == 3
    assert after == before


def test_a_real_run_processes_every_file_including_broken_ones(settings, temp_batch_folder):
    from PIL import Image

    from app.database.connection import connect

    Image.new("RGB", (80, 60), (90, 90, 90)).save(temp_batch_folder / "good.png")
    (temp_batch_folder / "empty.png").write_bytes(b"")
    (temp_batch_folder / "text.png").write_text("definitely not a png")

    info = batch_service.start_run(
        temp_batch_folder.name, "steel", settings=settings, synchronous=True
    )
    connection = connect(settings.db_file)
    try:
        totals = batch_service.compute_totals(connection, info["batch_run_id"])
    finally:
        connection.close()

    assert totals["processed"] == 3
    assert totals["failed"] == 2
    assert totals["clean"] + totals["regions_found"] == 1
