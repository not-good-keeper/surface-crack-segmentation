"""Capture screenshots of every screen, at desktop and phone widths.

Usage:
    python -m scripts.capture_screenshots

Starts the application against the configured database, drives a real Chromium via
Playwright and writes PNGs into docs/screenshots/.  Requires the browser binary:

    python -m playwright install chromium
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

from app.config import get_settings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"

DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}

SHOTS = [
    ("01-live-inspection", "/live", DESKTOP, True),
    ("02-region-detail", "/regions", DESKTOP, True),
    ("03-batch-run-report", "/batch", DESKTOP, True),
    ("04-inspection-history", "/history", DESKTOP, True),
    ("05-materials-thresholds", "/materials", DESKTOP, True),
    ("06-system-status", "/status", DESKTOP, True),
    ("07-history-filtered", "/history?material=steel&status=regions_found", DESKTOP, True),
    ("08-history-empty-state", "/history?product_id=no-such-product", DESKTOP, False),
    ("09-region-side-by-side", "/regions?mode=side_by_side&zoom=3", DESKTOP, True),
    ("10-batch-only-with-regions", "/batch?only_with_regions=true", DESKTOP, True),
    ("11-live-phone", "/live", PHONE, True),
    ("12-history-phone", "/history", PHONE, True),
    ("13-region-detail-phone", "/regions", PHONE, True),
    ("14-status-phone", "/status", PHONE, False),
    ("15-materials-phone", "/materials", PHONE, True),
    ("16-batch-phone", "/batch", PHONE, True),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int):
    import uvicorn

    from app.main import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    )

    def serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    deadline = time.time() + 120
    while not server.started and time.time() < deadline:
        time.sleep(0.2)
    if not server.started:
        raise RuntimeError("the application did not start")
    return server, thread


def capture() -> list[Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright && python -m playwright install chromium")
        return []

    settings = get_settings()
    settings.ensure_dirs()

    port = free_port()
    server, thread = start_server(port)
    base = f"http://127.0.0.1:{port}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()

            # Put a regions-found result at the station first: that is the state the
            # Live screen is designed around. The clean and failure states get their
            # own captures below.
            setup = browser.new_context(viewport=DESKTOP)
            setup_page = setup.new_page()
            setup_page.goto(f"{base}/live")
            setup_page.evaluate("fetch('/api/demo/next?force=mixed', {method: 'POST'})")
            setup_page.wait_for_timeout(1200)
            setup.close()
            for name, path, viewport, full_page in SHOTS:
                context = browser.new_context(viewport=viewport, device_scale_factor=1)
                page = context.new_page()
                page.goto(f"{base}{path}")
                page.wait_for_load_state("networkidle")
                # Stop the live screen advancing mid-capture.
                if page.locator("#auto-advance").count():
                    page.uncheck("#auto-advance")
                page.wait_for_timeout(400)

                target = OUTPUT_DIR / f"{name}.png"
                page.screenshot(path=str(target), full_page=full_page)
                written.append(target)
                print(f"  {target.name:<34} {viewport['width']}x{viewport['height']}  {path}")
                context.close()

            # The other two result states.
            context = browser.new_context(viewport=DESKTOP)
            page = context.new_page()
            page.goto(f"{base}/live")
            page.evaluate("fetch('/api/demo/next?force=clean', {method: 'POST'})")
            page.wait_for_timeout(1000)
            page.goto(f"{base}/live")
            page.wait_for_load_state("networkidle")
            if page.locator("#auto-advance").count():
                page.uncheck("#auto-advance")
            target = OUTPUT_DIR / "19-live-no-defects-found.png"
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(f"  {target.name:<34} NO DEFECTS FOUND state")

            page.evaluate("fetch('/api/demo/next?force=acquisition_failure', {method: 'POST'})")
            page.wait_for_timeout(1000)
            page.goto(f"{base}/live")
            page.wait_for_load_state("networkidle")
            if page.locator("#auto-advance").count():
                page.uncheck("#auto-advance")
            target = OUTPUT_DIR / "20-live-could-not-process.png"
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(f"  {target.name:<34} COULD NOT PROCESS state")
            context.close()

            # Restore a regions-found result before the station-fault captures.
            context = browser.new_context(viewport=DESKTOP)
            page = context.new_page()
            page.goto(f"{base}/live")
            page.evaluate("fetch('/api/demo/next?force=mixed', {method: 'POST'})")
            page.wait_for_timeout(1000)
            context.close()

            # Two states that need the station to be faulted first.
            context = browser.new_context(viewport=DESKTOP)
            page = context.new_page()
            page.goto(f"{base}/live")
            page.evaluate("fetch('/api/status/check?simulate=camera_offline', {method: 'POST'})")
            page.wait_for_timeout(800)
            page.goto(f"{base}/status")
            page.wait_for_load_state("networkidle")
            target = OUTPUT_DIR / "17-status-check-failing.png"
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(f"  {target.name:<34} station fault simulated")

            page.goto(f"{base}/live")
            page.wait_for_load_state("networkidle")
            target = OUTPUT_DIR / "18-live-check-station.png"
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(f"  {target.name:<34} CHECK STATION state")

            page.evaluate("fetch('/api/status/check?simulate=none', {method: 'POST'})")
            page.wait_for_timeout(500)
            context.close()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    return written


def main() -> None:
    print(f"Writing screenshots to {OUTPUT_DIR}")
    written = capture()
    print(f"\n{len(written)} screenshots written.")
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
