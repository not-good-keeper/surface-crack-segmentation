"""API routes: live, demo advance, listing, detail, region detail, structured errors."""

from __future__ import annotations

import pytest


def test_live_payload_has_health_and_inspection(client):
    payload = client.get("/api/live").json()
    assert payload["station"]
    assert payload["provider"] == "mock"
    assert "health" in payload and "overall" in payload["health"]
    assert payload["inspection"]["summary"]["headline"]


def test_live_payload_carries_no_score_field(client):
    body = client.get("/api/live").text.lower()
    for forbidden in ("confidence", "probability", '"score"'):
        assert forbidden not in body


def test_demo_next_creates_a_new_inspection(client):
    before = client.get("/api/live").json()["inspection"]["inspection_id"]
    response = client.post("/api/demo/next")
    assert response.status_code == 200
    after = response.json()["inspection_id"]
    assert after != before
    assert client.get("/api/live").json()["inspection"]["inspection_id"] == after


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("clean", "NO DEFECTS FOUND"),
        ("acquisition_failure", "COULD NOT PROCESS"),
        ("processing_failure", "COULD NOT PROCESS"),
    ],
)
def test_demo_next_can_force_each_state(client, kind, expected):
    payload = client.post(f"/api/demo/next?force={kind}").json()
    assert payload["inspection"]["summary"]["headline"] == expected


def test_forced_failure_is_not_reported_as_clean(client):
    payload = client.post("/api/demo/next?force=acquisition_failure").json()["inspection"]
    assert payload["status"] == "acquisition_failure"
    assert payload["error_code"]
    assert payload["summary"]["headline"] != "NO DEFECTS FOUND"


def test_inspection_list_is_paginated(client):
    first = client.get("/api/inspections?page=1&page_size=5").json()
    second = client.get("/api/inspections?page=2&page_size=5").json()
    assert len(first["items"]) == 5
    assert first["page_count"] >= 2
    assert {i["inspection_id"] for i in first["items"]} & {
        i["inspection_id"] for i in second["items"]
    } == set()


def test_inspection_detail_includes_model_and_profile(client):
    listing = client.get("/api/inspections?status=regions_found&page_size=1").json()
    inspection_id = listing["items"][0]["inspection_id"]
    payload = client.get(f"/api/inspections/{inspection_id}").json()

    assert payload["model"]["artefact_sha256"]
    assert payload["profile"]["crack_threshold"] is not None
    assert payload["profile"]["minimum_area_px"] is not None
    assert payload["image_sha256"]
    assert payload["regions"]


def test_region_detail_gives_navigation_indices(client):
    listing = client.get("/api/inspections?status=regions_found&page_size=20").json()
    target = next(i for i in listing["items"] if i["region_count"] >= 2)
    payload = client.get(f"/api/inspections/{target['inspection_id']}/regions/1").json()

    assert payload["region"]["region_index"] == 1
    assert payload["prev_index"] is None
    assert payload["next_index"] == 2
    assert "widest inscribable point" in payload["measurement_note"]


def test_region_detail_reports_geometry_fields(client):
    listing = client.get("/api/inspections?status=regions_found&page_size=1").json()
    payload = client.get(f"/api/inspections/{listing['items'][0]['inspection_id']}/regions/1").json()
    region = payload["region"]
    for field in ("area_px", "length_px", "max_width_px", "bbox", "centroid"):
        assert field in region
    assert set(region["bbox"]) == {"x", "y", "width", "height"}


def test_missing_inspection_returns_a_structured_error(client):
    response = client.get("/api/inspections/999999")
    assert response.status_code == 404
    assert "error" in response.json()


def test_missing_region_returns_404(client):
    listing = client.get("/api/inspections?status=regions_found&page_size=1").json()
    response = client.get(f"/api/inspections/{listing['items'][0]['inspection_id']}/regions/999")
    assert response.status_code == 404


def test_model_endpoint_reports_the_contract(client):
    payload = client.get("/api/model").json()
    assert payload["input_name"] == "input"
    assert payload["input_shape"] == [1, 3, 256, 256]
    assert payload["output_name"] == "logits"
    assert payload["channel_order"] == "RGB"
    assert "background" in payload["output_taxonomy"][0]
    assert "ARM" in payload["arm_note"]


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"
