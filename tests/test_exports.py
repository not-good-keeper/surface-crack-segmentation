"""CSV and JSON exports (FR-19 / T-16).

Exported totals are recalculated independently and compared against the displayed ones.
"""

from __future__ import annotations

import csv
import io
import json

from app.repositories import batch_repository as batches
from app.repositories.inspection_repository import HistoryFilters
from app.services import batch_service, export_service


def read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_csv_has_one_row_per_region_plus_one_per_non_region_inspection(conn):
    run = batches.recent(conn, limit=1)[0]
    filters = batch_service.batch_filters(run["batch_run_id"])
    rows = read_csv(export_service.to_csv(conn, filters))

    totals = batch_service.compute_totals(conn, run["batch_run_id"])
    expected = totals["regions_total"] + totals["clean"] + totals["failed"]
    assert len(rows) == expected


def test_csv_totals_recalculate_to_the_displayed_totals(conn):
    run = batches.recent(conn, limit=1)[0]
    filters = batch_service.batch_filters(run["batch_run_id"])
    rows = read_csv(export_service.to_csv(conn, filters))
    totals = batch_service.compute_totals(conn, run["batch_run_id"])

    distinct = {r["inspection_id"] for r in rows}
    assert len(distinct) == totals["processed"]

    clean_ids = {r["inspection_id"] for r in rows if r["status"] == "clean"}
    assert len(clean_ids) == totals["clean"]

    failure_ids = {
        r["inspection_id"] for r in rows
        if r["status"] in ("acquisition_failure", "processing_failure")
    }
    assert len(failure_ids) == totals["failed"]

    region_rows = [r for r in rows if r["region_index"]]
    assert len(region_rows) == totals["regions_total"]


def test_csv_carries_provenance_on_every_row(conn):
    rows = read_csv(export_service.to_csv(conn, HistoryFilters(page_size=500)))
    for row in rows[:25]:
        assert row["model_version"]
        assert row["model_sha256"]
        assert row["profile_version"]
        assert row["crack_threshold"]
        assert row["captured_at"]


def test_csv_failure_rows_carry_a_status_and_no_regions(conn):
    rows = read_csv(export_service.to_csv(conn, HistoryFilters(status="acquisition_failure", page_size=500)))
    assert rows
    for row in rows:
        assert row["status"] == "acquisition_failure"
        assert row["region_index"] == ""
        assert row["error_code"]


def test_csv_never_labels_a_failure_as_clean(conn):
    rows = read_csv(export_service.to_csv(conn, HistoryFilters(page_size=1000)))
    for row in rows:
        if row["error_code"]:
            assert row["status"] != "clean"


def test_json_export_totals_match_the_repository(conn):
    run = batches.recent(conn, limit=1)[0]
    filters = batch_service.batch_filters(run["batch_run_id"])
    payload = json.loads(export_service.to_json(conn, filters))
    totals = batch_service.compute_totals(conn, run["batch_run_id"])

    assert payload["count"] == totals["processed"]
    assert payload["totals"]["clean"] == totals["clean"]
    assert payload["totals"]["regions_found"] == totals["regions_found"]
    assert sum(len(i["regions"]) for i in payload["inspections"]) == totals["regions_total"]


def test_json_export_includes_geometry_and_thresholds(conn):
    payload = json.loads(export_service.to_json(conn, HistoryFilters(status="regions_found", page_size=10)))
    inspection = payload["inspections"][0]
    assert inspection["profile"]["crack_threshold"] is not None
    region = inspection["regions"][0]
    for field in ("area_px", "length_px", "max_width_px", "bbox", "centroid"):
        assert field in region


def test_exports_contain_no_score_or_verdict_wording(conn):
    body = export_service.to_csv(conn, HistoryFilters(page_size=200)).lower()
    body += export_service.to_json(conn, HistoryFilters(page_size=50)).lower()
    for forbidden in ("confidence", "probability", "accept", "reject", "% confident"):
        assert forbidden not in body


def test_csv_endpoint_sets_a_download_header(client):
    response = client.get("/api/inspections/export.csv?status=clean")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]


def test_batch_export_endpoints_match_the_screen(client, conn):
    run = batches.recent(conn, limit=1)[0]
    run_id = run["batch_run_id"]
    api = client.get(f"/api/batches/{run_id}").json()
    rows = read_csv(client.get(f"/api/batches/{run_id}/export.csv").text)
    assert len({r["inspection_id"] for r in rows}) == api["totals"]["processed"]

    payload = client.get(f"/api/batches/{run_id}/export.json").json()
    assert payload["count"] == api["totals"]["processed"]


def test_filtered_export_matches_the_filtered_screen(client, conn):
    run = batches.recent(conn, limit=1)[0]
    run_id = run["batch_run_id"]
    api = client.get(f"/api/batches/{run_id}?only_with_regions=true").json()
    rows = read_csv(client.get(f"/api/batches/{run_id}/export.csv?only_with_regions=true").text)
    assert len({r["inspection_id"] for r in rows}) == api["filtered_totals"]["processed"]
    assert all(r["status"] == "regions_found" for r in rows)


def test_history_export_respects_the_filter(client, conn):
    api = client.get("/api/inspections?material=steel&status=clean&page_size=200").json()
    rows = read_csv(client.get("/api/inspections/export.csv?material=steel&status=clean").text)
    assert len({r["inspection_id"] for r in rows}) == api["total"]
