"""Camera capture and photo upload.

The capture path is the one place an operator hands the system an image it has never
seen, so the properties worth pinning are the ones that would otherwise be discovered
by a user with a real photograph:

* a capture is stored like any other inspection, so it appears in History and exports
* a file that is not an image becomes an acquisition failure, never `clean`
* the same photograph submitted twice produces the same mock result
* a failing station check stops capture, exactly as it stops the line
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services import inspection_service, status_service


def png_bytes(width: int = 240, height: int = 180, colour: tuple[int, int, int] = (128, 132, 136)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def post_capture(client, payload: bytes, *, name: str = "frame.png", source: str = "camera",
                 material: str = "steel", product_id: str | None = None):
    data = {"material": material, "source": source}
    if product_id:
        data["product_id"] = product_id
    return client.post(
        "/api/inspections/capture",
        files={"file": (name, payload, "image/png")},
        data=data,
    )


# -- the screen ---------------------------------------------------------------
def test_capture_screen_loads(client):
    response = client.get("/capture")
    assert response.status_code == 200
    body = response.text
    assert "Device camera" in body
    assert "Photo from this device" in body


def test_capture_is_reachable_from_every_screen(client):
    for path in ("/live", "/regions", "/batch", "/history", "/materials", "/status"):
        assert 'href="/capture"' in client.get(path).text


def test_mock_mode_says_the_regions_are_generated(client, settings):
    """The easiest mistake this application could invite, guarded explicitly.

    An operator photographs a genuinely cracked part, sees regions drawn on their own
    photograph, and concludes the model found them. In mock mode nothing found
    anything, and the screen has to say so before the result appears.
    """
    assert settings.is_mock
    body = client.get("/capture").text
    assert "generated, not measured" in body
    assert "no model is running" in body


# -- storing a capture --------------------------------------------------------
def test_a_capture_is_stored_and_shows_up_in_history(client):
    response = post_capture(client, png_bytes(), product_id="capture-test/item-1")
    assert response.status_code == 200
    payload = response.json()

    inspection_id = payload["inspection_id"]
    assert payload["inspection"]["status"] in ("regions_found", "clean")
    assert payload["generated"] is True  # mock mode

    detail = client.get(f"/api/inspections/{inspection_id}")
    assert detail.status_code == 200
    assert detail.json()["product_id"] == "capture-test/item-1"

    listed = client.get("/api/inspections", params={"product_id": "capture-test/item-1"}).json()
    assert listed["total"] >= 1


def test_a_product_id_is_derived_from_the_filename_when_omitted(client):
    payload = post_capture(client, png_bytes(241, 181), name="IMG_4021.png", source="upload").json()
    assert payload["inspection"]["product_id"].startswith("upload/")
    assert "IMG_4021" in payload["inspection"]["product_id"]


def test_the_source_is_restricted_to_camera_or_upload(client):
    response = post_capture(client, png_bytes(), source="ftp")
    assert response.status_code == 422


# -- failure states stay distinct ---------------------------------------------
def test_a_file_that_is_not_an_image_is_an_acquisition_failure(client):
    response = client.post(
        "/api/inspections/capture",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
        data={"source": "upload"},
    )
    assert response.status_code == 200
    inspection = response.json()["inspection"]
    # The distinction the whole application is built around: a broken file is not a
    # good part.
    assert inspection["status"] == "acquisition_failure"
    assert inspection["status"] != "clean"
    assert inspection["error_code"]
    assert inspection["region_count"] == 0


def test_an_empty_file_is_an_acquisition_failure(client):
    response = client.post(
        "/api/inspections/capture",
        files={"file": ("empty.png", b"", "image/png")},
        data={"source": "upload"},
    )
    assert response.json()["inspection"]["status"] == "acquisition_failure"


def test_an_oversized_capture_is_refused(client):
    oversize = b"\x89PNG\r\n\x1a\n" + b"\x00" * inspection_service.CAPTURE_MAX_BYTES
    response = client.post(
        "/api/inspections/capture",
        files={"file": ("huge.png", oversize, "image/png")},
        data={"source": "upload"},
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_the_stated_limit_matches_the_limit_enforced(client):
    """The screen offers a size; the error refuses at one. They must be the same."""
    body = client.get("/capture").text
    stated = f"up to {inspection_service.CAPTURE_MAX_BYTES // (1024 * 1024)} MB"
    assert stated in body


# -- determinism --------------------------------------------------------------
def test_the_same_photograph_gives_the_same_mock_result(client):
    frame = png_bytes(300, 200, (140, 140, 145))
    first = post_capture(client, frame).json()["inspection"]
    second = post_capture(client, frame).json()["inspection"]

    assert first["inspection_id"] != second["inspection_id"]  # two separate records
    assert first["status"] == second["status"]
    assert first["region_count"] == second["region_count"]
    assert [r["area_px"] for r in first["regions"]] == [r["area_px"] for r in second["regions"]]


# -- a failing station stops capture ------------------------------------------
def test_capture_is_blocked_while_a_station_check_fails(client, reset_simulation):
    status_service.set_simulation("model_hash_mismatch")
    try:
        response = post_capture(client, png_bytes())
        assert response.status_code == 409
        assert "stopped" in response.json()["detail"].lower()
    finally:
        status_service.set_simulation("none")


# -- no forbidden wording -----------------------------------------------------
@pytest.mark.parametrize("forbidden", ["% confident", "confidence:", "probability:"])
def test_the_capture_screen_states_no_confidence(client, forbidden):
    assert forbidden.lower() not in client.get("/capture").text.lower()
