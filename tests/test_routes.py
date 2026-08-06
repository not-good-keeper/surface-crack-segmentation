"""Page routes: every screen loads, navigation is marked, the health strip persists."""

from __future__ import annotations

import pytest

PAGES = ["/live", "/regions", "/batch", "/history", "/materials", "/status"]


@pytest.mark.parametrize("path", PAGES)
def test_every_page_returns_html(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<main" in response.text


def test_root_redirects_to_live(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/live"


@pytest.mark.parametrize("path", PAGES)
def test_navigation_marks_the_current_screen(client, path):
    body = client.get(path).text
    # Scoped to the nav: pagination legitimately uses aria-current="page" too.
    nav = body.split('<nav class="nav"')[1].split("</nav>")[0]
    assert nav.count('aria-current="page"') == 1
    assert nav.count("nav__link--current") == 1
    for item in ("Live", "Regions", "Batch", "History", "Materials", "Status"):
        assert f">{item}</span>" in body


@pytest.mark.parametrize("path", PAGES)
def test_health_strip_appears_on_every_screen(client, path):
    body = client.get(path).text
    assert 'aria-label="Station health"' in body
    assert "Camera" in body and "Disk" in body


@pytest.mark.parametrize("path", PAGES)
def test_every_page_has_a_skip_link_and_one_h1(client, path):
    body = client.get(path).text
    assert 'class="skip-link"' in body
    assert body.count("<h1") == 1


def test_live_shows_a_result_banner(client):
    body = client.get("/live").text
    assert 'id="result-banner"' in body
    assert 'aria-live="polite"' in body


def test_live_shows_the_provisional_class_notice(client):
    assert "provisional" in client.get("/live").text.lower()


def test_region_detail_states_the_max_width_limitation(client):
    body = client.get("/regions").text
    assert "widest inscribable point" in body
    assert "widest visual extent" in body


def test_region_detail_supports_image_modes_and_zoom(client):
    response = client.get("/regions?mode=side_by_side&zoom=4")
    assert response.status_code == 200
    assert "Side by side" in response.text


def test_inspection_detail_page_loads(client):
    listing = client.get("/api/inspections?status=regions_found&page_size=1").json()
    inspection_id = listing["items"][0]["inspection_id"]
    response = client.get(f"/inspections/{inspection_id}")
    assert response.status_code == 200
    assert "Provenance" in response.text


def test_unknown_page_returns_a_helpful_404(client):
    response = client.get("/not-a-screen")
    assert response.status_code == 404
    assert "PAGE NOT FOUND" in response.text


def test_static_assets_are_served_locally(client):
    for asset in (
        "/static/css/base.css",
        "/static/css/layout.css",
        "/static/css/components.css",
        "/static/css/responsive.css",
        "/static/js/live.js",
        "/static/icons/favicon.svg",
    ):
        assert client.get(asset).status_code == 200


def test_no_page_references_an_external_origin(client):
    """Offline by construction: nothing may point at a CDN or a web font."""
    for path in PAGES:
        body = client.get(path).text
        for marker in ("http://", "https://", "//cdn", "fonts.googleapis"):
            assert marker not in body, f"{path} references an external origin: {marker}"
