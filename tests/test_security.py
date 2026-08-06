"""Path traversal, invalid files and media access."""

from __future__ import annotations

import pytest

from app.services import batch_service, inspection_service


@pytest.mark.parametrize(
    "attempt",
    [
        "../../../../etc/passwd",
        "/etc/passwd",
        "..%2f..%2fetc",
        "....//....//etc",
        "steel/../../../root",
    ],
)
def test_batch_api_refuses_traversal(client, attempt):
    response = client.post("/api/batches", json={"source_folder": attempt, "material": "steel"})
    assert response.status_code == 400
    assert "outside" in response.json()["detail"] or "not a folder" in response.json()["detail"]


def test_batch_api_refuses_an_absolute_path_outside_the_root(client):
    response = client.post("/api/batches", json={"source_folder": "/tmp", "material": "steel"})
    assert response.status_code == 400


def test_media_route_takes_no_path_from_the_request(client):
    """There is no endpoint that accepts a filesystem path, so there is nothing to traverse."""
    for attempt in (
        "/media/inspection/1/../../../etc/passwd",
        "/media/inspection/1/source/../../etc",
        "/media/inspection/abc/source",
    ):
        assert client.get(attempt).status_code in (404, 422)


def test_media_rejects_an_unknown_image_kind(client):
    assert client.get("/media/inspection/1/secrets").status_code == 404


def test_media_rejects_an_unknown_inspection(client):
    assert client.get("/media/inspection/999999/overlay").status_code == 404


def test_stored_paths_outside_the_data_roots_are_refused(settings):
    """Even a path that came from the database is checked before it is read."""
    row = {"source_image_path": "/etc/passwd", "overlay_image_path": "/etc/passwd"}
    assert inspection_service.image_path_for(row, "source", settings) is None
    assert inspection_service.image_path_for(row, "overlay", settings) is None


def test_stored_paths_inside_the_data_roots_resolve(conn, settings):
    row = conn.execute(
        "SELECT * FROM inspection WHERE overlay_image_path IS NOT NULL LIMIT 1"
    ).fetchone()
    assert inspection_service.image_path_for(dict(row), "overlay", settings) is not None


def test_region_crop_rejects_an_out_of_range_zoom(client, conn):
    row = conn.execute("SELECT inspection_id FROM inspection WHERE region_count > 0 LIMIT 1").fetchone()
    inspection_id = row["inspection_id"]
    assert client.get(f"/media/inspection/{inspection_id}/region/1?zoom=99").status_code == 422
    assert client.get(f"/media/inspection/{inspection_id}/region/1?mode=evil").status_code == 422


def test_oversized_files_are_skipped_by_the_batch_scanner(settings, temp_batch_folder, monkeypatch):
    big = temp_batch_folder / "huge.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 2048)
    monkeypatch.setattr(batch_service, "MAX_FILE_BYTES", 512)
    assert batch_service.image_files(temp_batch_folder) == []


def test_a_decompression_bomb_is_refused_by_the_provider(settings, monkeypatch):
    """A declared size beyond the decode limit is rejected before it is expanded."""
    import io

    from PIL import Image

    from app.providers import mock_provider
    from app.providers.mock_provider import MockInspectionProvider

    monkeypatch.setattr(mock_provider, "MAX_DECODED_PIXELS", 100)
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (10, 10, 10)).save(buffer, format="PNG")

    result = MockInspectionProvider(settings).inspect(None, buffer.getvalue(), "p", "steel", key="bomb")
    assert result.status == "acquisition_failure"
    assert result.error_code == "oversized_image"


def test_demo_control_is_refused_when_demo_mode_is_off(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    assert client.post("/api/demo/next").status_code == 403


def test_fault_simulation_is_refused_in_real_mode(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "inspection_provider", "real")
    assert client.post("/api/status/check?simulate=camera_offline").status_code == 403
