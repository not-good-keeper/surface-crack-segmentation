"""Analytics dashboard: the cross-session overview and one dashboard per session.

A "session" is a batch run - the one place the schema already groups many inspections
under a single id. These tests mirror the conventions in test_routes.py and
test_content_rules.py: every screen loads, the nav marks it current, the health strip
persists, and nothing on it points at an external origin.
"""

from __future__ import annotations

import re


def test_analytics_overview_loads(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<main" in response.text


def test_analytics_overview_marks_the_current_screen(client):
    body = client.get("/analytics").text
    nav = body.split('<nav class="nav"')[1].split("</nav>")[0]
    assert nav.count('aria-current="page"') == 1
    assert nav.count("nav__link--current") == 1
    assert ">Analytics</span>" in body


def test_analytics_overview_shows_the_health_strip(client):
    body = client.get("/analytics").text
    assert 'aria-label="Station health"' in body


def test_analytics_overview_lists_seeded_sessions(client):
    body = client.get("/analytics").text
    assert "Sessions" in body
    assert 'class="session-tile"' in body
    assert "/analytics/1" in body


def test_analytics_overview_renders_charts_with_titles(client):
    """Charts carry an accessible value via <title>, not colour alone."""
    body = client.get("/analytics").text
    assert 'class="chart"' in body
    assert "<title>" in body


def test_analytics_session_dashboard_loads_for_a_seeded_run(client):
    response = client.get("/analytics/1")
    assert response.status_code == 200
    assert "Session 1 Analytics" in response.text
    assert 'class="chart"' in response.text


def test_analytics_session_dashboard_links_back_to_the_batch_report(client):
    body = client.get("/analytics/1").text
    assert '/batch?batch_run_id=1' in body


def test_analytics_session_dashboard_totals_match_the_batch_report(client, conn):
    from app.services import analytics_service, batch_service

    dash = analytics_service.session_dashboard(conn, 1)
    totals = batch_service.compute_totals(conn, 1)
    assert dash["totals"] == totals


def test_analytics_unknown_session_returns_404(client):
    response = client.get("/analytics/999999")
    assert response.status_code == 404


def test_analytics_pages_reference_no_external_origin(client):
    for path in ("/analytics", "/analytics/1"):
        body = client.get(path).text
        for marker in ("http://", "https://", "//cdn", "fonts.googleapis"):
            assert marker not in body, f"{path} references an external origin: {marker}"


def test_analytics_pages_contain_no_confidence_wording(client):
    forbidden = ["% confident", "confidence:", "probability:", "certainty"]
    for path in ("/analytics", "/analytics/1"):
        lowered = client.get(path).text.lower()
        for phrase in forbidden:
            assert phrase not in lowered, f"{path} contains forbidden phrase: {phrase}"


def test_analytics_pages_present_no_accept_reject_verdict(client):
    verdicts = re.compile(r"\b(accept|accepted|reject|rejected|pass/fail|failed part)\b", re.IGNORECASE)
    for path in ("/analytics", "/analytics/1"):
        body = client.get(path).text
        assert not verdicts.findall(body), f"{path} contains verdict wording"


def test_analytics_class_labels_are_marked_provisional(client):
    assert "provisional" in client.get("/analytics").text.lower()
    assert "provisional" in client.get("/analytics/1").text.lower()


def test_status_mix_legend_never_relies_on_colour_alone(client):
    body = client.get("/analytics/1").text
    for label in ("Clean", "Regions found"):
        assert label in body
