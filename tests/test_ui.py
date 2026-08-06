"""Browser tests (Playwright).

Skipped automatically when a browser is not installed, so the suite still runs on a
machine that only has the Python dependencies.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_api.sync_playwright

VIEWPORTS = {
    "desktop_1440": (1440, 900),
    "laptop_1024": (1024, 768),
    "tablet_768": (768, 1024),
    "phone_390": (390, 844),
    "phone_360": (360, 780),
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(seeded):
    """A real uvicorn server on a free port.

    Depends on ``seeded`` so the database already exists: otherwise application
    start-up would seed it, which generates several hundred images and takes minutes.
    The server runs on its own event loop in a worker thread, because uvicorn's
    ``run()`` installs signal handlers that only work on the main thread.
    """
    import asyncio

    import uvicorn

    from app.main import create_app

    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    )

    def serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    if not server.started:
        pytest.skip("the test server did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser(live_server):
    try:
        with sync_playwright() as playwright:
            instance = playwright.chromium.launch()
            yield instance
            instance.close()
    except Exception as exc:  # browser binary not installed
        pytest.skip(f"no browser available: {exc}")


@pytest.fixture()
def page(browser):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    yield page
    context.close()


# -- navigation ---------------------------------------------------------------
def test_navigation_reaches_every_screen_in_one_click(page, live_server):
    page.goto(f"{live_server}/live")
    for label, heading in [
        ("Regions", "Region Detail"),
        ("Batch", "Batch Run and Report"),
        ("History", "Inspection History"),
        ("Materials", "Materials and Thresholds"),
        ("Status", "System Status"),
        ("Live", "Live Inspection"),
    ]:
        page.click(f".nav__link:has-text('{label}')")
        page.wait_for_load_state("networkidle")
        assert heading in page.inner_text("h1")


def test_current_screen_is_highlighted(page, live_server):
    page.goto(f"{live_server}/history")
    current = page.locator(".nav__link--current")
    assert current.count() == 1
    assert current.inner_text().strip() == "History"


def test_health_strip_is_present_on_every_screen(page, live_server):
    for path in ("/live", "/regions", "/batch", "/history", "/materials", "/status"):
        page.goto(f"{live_server}{path}")
        assert page.locator("[aria-label='Station health']").is_visible()


# -- live updates -------------------------------------------------------------
def test_live_result_updates_when_the_station_advances(page, live_server):
    page.goto(f"{live_server}/live")
    page.uncheck("#auto-advance")
    first = page.get_attribute("#live-root", "data-inspection-id")

    page.click("#next-inspection")
    page.wait_for_function(
        "id => document.getElementById('live-root').getAttribute('data-inspection-id') !== id",
        arg=first,
        timeout=15000,
    )
    assert page.get_attribute("#live-root", "data-inspection-id") != first
    assert page.locator("#result-banner").inner_text().strip() != ""


def test_live_banner_states_are_distinct(page, live_server):
    page.goto(f"{live_server}/live")
    page.uncheck("#auto-advance")

    headlines = {}
    for kind in ("clean", "acquisition_failure"):
        page.evaluate(f"fetch('/api/demo/next?force={kind}', {{method: 'POST'}})")
        page.wait_for_timeout(1500)
        page.reload()
        headlines[kind] = page.locator("#result-banner .banner__headline").inner_text().strip()

    assert headlines["clean"] == "NO DEFECTS FOUND"
    assert headlines["acquisition_failure"] == "COULD NOT PROCESS"


# -- region navigation --------------------------------------------------------
def test_region_previous_and_next_move_through_regions(page, live_server):
    page.goto(f"{live_server}/history?status=regions_found")
    page.click("tbody tr:first-child a")
    page.wait_for_load_state("networkidle")
    page.click("text=Open region detail")
    page.wait_for_load_state("networkidle")

    current = page.locator(".region-list__item--current").first.inner_text()
    if page.locator("#next-region").count():
        page.click("#next-region")
        page.wait_for_load_state("networkidle")
        assert page.locator(".region-list__item--current").first.inner_text() != current
        page.click("#prev-region")
        page.wait_for_load_state("networkidle")
        assert page.locator(".region-list__item--current").first.inner_text() == current


def test_image_mode_toggle_switches_the_view(page, live_server):
    page.goto(f"{live_server}/regions")
    if page.locator("#region-crop").count() == 0:
        pytest.skip("no region available")
    page.click(".chip:has-text('Side by side')")
    page.wait_for_load_state("networkidle")
    assert page.locator(".viewer--split").count() == 1
    assert page.locator(".viewer__pane").count() == 2


def test_zoom_control_changes_the_crop_request(page, live_server):
    page.goto(f"{live_server}/regions")
    if page.locator("#region-crop").count() == 0:
        pytest.skip("no region available")
    page.click(".chip:has-text('6×')")
    page.wait_for_load_state("networkidle")
    assert "zoom=6" in page.get_attribute("#region-crop", "src")


# -- history ------------------------------------------------------------------
def test_history_filtering_narrows_the_table(page, live_server):
    page.goto(f"{live_server}/history")
    before = page.inner_text("#results-heading")
    page.select_option("#status", "clean")
    page.click("button:has-text('Apply')")
    page.wait_for_load_state("networkidle")

    assert "status=clean" in page.url
    rows = page.locator("tbody tr")
    for index in range(rows.count()):
        assert "clean" in rows.nth(index).inner_text()
    assert page.inner_text("#results-heading") != before or rows.count() > 0


def test_history_empty_state(page, live_server):
    page.goto(f"{live_server}/history?product_id=nothing-matches-this")
    assert "No inspections match these filters" in page.inner_text("main")


def test_filters_survive_going_back(page, live_server):
    page.goto(f"{live_server}/history?material=steel&status=regions_found")
    page.click("tbody tr:first-child a")
    page.wait_for_load_state("networkidle")
    page.go_back()
    page.wait_for_load_state("networkidle")
    assert page.input_value("#status") == "regions_found"
    assert page.input_value("#material") == "steel"


# -- batch --------------------------------------------------------------------
def test_batch_summary_cards_reconcile_with_the_table(page, live_server):
    page.goto(f"{live_server}/batch")
    processed = int(page.inner_text("#card-processed").replace(",", ""))
    rows = page.locator("table tbody tr").first
    assert processed >= 1
    assert rows.count() >= 1

    regions = int(page.inner_text("#card-regions").replace(",", ""))
    clean = int(page.inner_text("#card-clean").replace(",", ""))
    failed = int(page.inner_text("#card-failed").replace(",", ""))
    assert regions + clean + failed == processed


def test_batch_dry_run_reports_without_processing(page, live_server):
    page.goto(f"{live_server}/batch")
    page.click("#dry-run")
    page.wait_for_load_state("networkidle")
    assert "batch_run_id=" in page.url


# -- responsive ---------------------------------------------------------------
@pytest.mark.parametrize("name,size", list(VIEWPORTS.items()))
def test_layout_has_no_horizontal_overflow(browser, live_server, name, size):
    width, height = size
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    try:
        for path in ("/live", "/regions", "/batch", "/history", "/materials", "/status"):
            page.goto(f"{live_server}{path}")
            page.wait_for_load_state("networkidle")
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            assert overflow <= 1, f"{path} overflows horizontally at {name} by {overflow}px"
    finally:
        context.close()


def test_navigation_becomes_a_top_bar_on_a_phone(browser, live_server):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    try:
        page.goto(f"{live_server}/live")
        nav = page.locator(".nav").bounding_box()
        main = page.locator("#main").bounding_box()
        # Stacked, not side by side.
        assert nav["y"] + nav["height"] <= main["y"] + 5
        assert nav["width"] > 300
    finally:
        context.close()


def test_result_state_is_near_the_top_on_a_phone(browser, live_server):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    try:
        page.goto(f"{live_server}/live")
        banner = page.locator("#result-banner").bounding_box()
        image = page.locator(".live__image").bounding_box()
        assert banner["y"] < image["y"], "the result must come before the image on a phone"
    finally:
        context.close()


def test_touch_targets_are_large_enough_on_a_phone(browser, live_server):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    try:
        page.goto(f"{live_server}/live")
        links = page.locator(".nav__link")
        for index in range(links.count()):
            box = links.nth(index).bounding_box()
            assert box["height"] >= 40, f"nav target only {box['height']}px tall"
    finally:
        context.close()


def test_tables_scroll_rather_than_shrink_on_a_phone(browser, live_server):
    context = browser.new_context(viewport={"width": 360, "height": 780})
    page = context.new_page()
    try:
        page.goto(f"{live_server}/history")
        font = page.evaluate(
            "() => getComputedStyle(document.querySelector('table td')).fontSize"
        )
        assert float(font.replace("px", "")) >= 13
        assert page.locator(".table-scroll").count() >= 1
    finally:
        context.close()


# -- static checks on the rendered DOM ---------------------------------------
def test_no_rendered_page_shows_a_percentage_score(browser, live_server):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    try:
        for path in ("/live", "/regions", "/batch", "/history", "/materials", "/status"):
            page.goto(f"{live_server}{path}")
            text = page.inner_text("body").lower()
            for phrase in ("% confident", "confidence", "probability"):
                assert phrase not in text, f"{path} shows '{phrase}'"
    finally:
        context.close()
