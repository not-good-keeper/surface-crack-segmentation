"""The rules about what must never appear on screen (T-06 / T-21).

These are the tests that would catch the interface quietly acquiring a confidence
percentage, an accept/reject verdict, or its own copy of the thresholds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

#: A threshold value being *stated*, in any of the three forms a copy would take: an
#: annotated dataclass field, a plain assignment, or a dict entry. The earlier pattern
#: (`crack_thresh\s*[:=]\s*[\d.]+`) matched none of them against the real declaration
#: `crack_thresh: float = 0.40`, so it could not have caught a copy written the same
#: way as the original — the most likely way one would appear.
THRESHOLD_DEFINITION = re.compile(
    r'(?:"crack_thresh"\s*:|crack_thresh\s*(?::\s*\w+\s*)?=)\s*[\d.]+'
)
TEMPLATES = sorted((APP_DIR / "templates").rglob("*.html"))
SCRIPTS = sorted((APP_DIR / "static" / "js").glob("*.js"))
STYLES = sorted((APP_DIR / "static" / "css").glob("*.css"))

PAGES = ["/live", "/regions", "/batch", "/history", "/materials", "/status"]

FORBIDDEN_PHRASES = [
    "% confident",
    "percent confident",
    "confidence:",
    "confidence :",
    "probability:",
    "probability of",
    "% confidence",
    "confidence score",
    "certainty",
]


def rendered_pages(client) -> dict[str, str]:
    bodies = {path: client.get(path).text for path in PAGES}
    listing = client.get("/api/inspections?status=regions_found&page_size=1").json()
    inspection_id = listing["items"][0]["inspection_id"]
    bodies["/inspections/{id}"] = client.get(f"/inspections/{inspection_id}").text
    bodies["/regions?full"] = client.get(f"/regions?inspection_id={inspection_id}&region=1").text
    return bodies


# -- no confidence wording ----------------------------------------------------
def test_no_rendered_page_contains_confidence_wording(client):
    for path, body in rendered_pages(client).items():
        lowered = body.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"{path} contains forbidden phrase: {phrase}"


def test_no_rendered_page_shows_a_score_as_a_percentage(client):
    """Percentages are allowed for disk and batch shares, never for a model score."""
    pattern = re.compile(r"(score|confiden|probabilit)[^<]{0,40}\d+\s*%", re.IGNORECASE)
    for path, body in rendered_pages(client).items():
        assert not pattern.search(body), f"{path} renders a score as a percentage"


def test_templates_contain_no_confidence_wording():
    for template in TEMPLATES:
        lowered = template.read_text().lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"{template.name}: {phrase}"


# -- no operator-facing verdict ----------------------------------------------
def test_no_page_presents_an_accept_or_reject_verdict(client):
    verdicts = re.compile(r"\b(accept|accepted|reject|rejected|pass/fail|failed part)\b", re.IGNORECASE)
    for path, body in rendered_pages(client).items():
        matches = verdicts.findall(body)
        assert not matches, f"{path} contains verdict wording: {set(matches)}"


def test_result_states_use_the_agreed_wording(client):
    body = client.get("/live").text
    assert any(
        phrase in body
        for phrase in ("REGIONS FOUND", "REGION FOUND", "NO DEFECTS FOUND", "COULD NOT PROCESS", "CHECK STATION")
    )


def test_clean_and_failure_never_share_wording(client):
    clean = client.post("/api/demo/next?force=clean").json()["inspection"]
    failed = client.post("/api/demo/next?force=acquisition_failure").json()["inspection"]

    assert clean["summary"]["headline"] == "NO DEFECTS FOUND"
    assert failed["summary"]["headline"] == "COULD NOT PROCESS"
    assert clean["summary"]["headline"] != failed["summary"]["headline"]
    assert failed["summary"]["detail"]  # names the reason


# -- no duplicated thresholds -------------------------------------------------
THRESHOLD_LITERALS = [
    (r"0\.40", "crack_thresh"),
    (r"0\.20", "scratch_thresh"),
    (r"\b24\b\s*(px)?", "min_area_px"),
    (r"crack_thresh\s*[=:]\s*[\d.]", "crack_thresh assignment"),
    (r"scratch_thresh\s*[=:]\s*[\d.]", "scratch_thresh assignment"),
    (r"min_area_px\s*[=:]\s*\d", "min_area_px assignment"),
    (r"min_skeleton_px\s*[=:]\s*\d", "min_skeleton_px assignment"),
]


@pytest.mark.parametrize("source", TEMPLATES + SCRIPTS, ids=lambda p: p.name)
def test_no_threshold_value_is_hardcoded_in_the_frontend(source):
    """The interface must read thresholds from the backend, never hold its own copy."""
    text = source.read_text()
    for pattern, label in THRESHOLD_LITERALS:
        assert not re.search(pattern, text), f"{source.name} hardcodes {label}"


def test_thresholds_on_screen_come_from_the_backend(client, conn):
    from app.services import material_service

    thresholds = material_service.thresholds(conn)["stored"]
    body = client.get("/materials").text
    assert str(thresholds["crack_threshold"]) in body
    assert str(thresholds["scratch_threshold"]) in body


def test_only_one_module_defines_the_threshold_values():
    """Exactly one module in the application states the numbers.

    That module is `app/profiles.py`: it is stdlib-only, so the interface can read the
    active thresholds in a deployment where cv2/numpy/scikit-image are not installed.
    `app/postprocess.py` re-exports the same object rather than restating the values,
    which is what keeps the pipeline and the interface from drifting apart (T-06).
    """
    definitions = [
        source.name
        for source in APP_DIR.rglob("*.py")
        # Test modules are exempt: they construct Profile with explicit values on
        # purpose, to drive the class-decision rule through cases the shipped
        # thresholds would not reach. That is exercising the rule, not restating it.
        if not source.name.startswith("test_")
        and re.search(THRESHOLD_DEFINITION, source.read_text())
    ]
    assert definitions == ["profiles.py"], (
        f"threshold values must be defined in profiles.py alone, found: {definitions}"
    )


# -- limitations stay visible -------------------------------------------------
def test_the_class_label_is_marked_provisional_where_it_appears(client):
    assert "provisional" in client.get("/live").text.lower()
    assert "provisional" in client.get("/regions").text.lower()


def test_region_detail_states_the_width_limitation(client):
    body = client.get("/regions").text
    assert "widest inscribable point" in body


def test_materials_screen_states_the_arm_limitation(client):
    assert "No ARM or phone latency measurement" in client.get("/materials").text


def test_overlay_is_described_as_source_resolution(client):
    assert "source resolution" in client.get("/live").text


# -- accessibility basics -----------------------------------------------------
def test_tables_use_header_cells(client):
    for path in ("/history", "/materials", "/batch"):
        body = client.get(path).text
        assert "<th scope=\"col\"" in body


def test_images_carry_alt_text(client):
    body = client.get("/live").text
    for match in re.finditer(r"<img\b[^>]*>", body):
        assert "alt=" in match.group(0)


def test_form_controls_have_labels(client):
    body = client.get("/history").text
    for control_id in ("date_from", "date_to", "material", "status", "class", "station", "product_id"):
        assert f'for="{control_id}"' in body
        assert f'id="{control_id}"' in body


def test_reduced_motion_is_respected():
    css = (APP_DIR / "static" / "css" / "base.css").read_text()
    assert "prefers-reduced-motion" in css


def test_focus_is_visible():
    css = (APP_DIR / "static" / "css" / "base.css").read_text()
    assert ":focus-visible" in css


def test_touch_targets_are_at_least_44px():
    css = "\n".join(path.read_text() for path in STYLES)
    assert "--touch: 44px" in css
    assert "min-height: var(--touch)" in css


def test_status_is_never_carried_by_colour_alone(client):
    """Every status badge carries a word, not just a fill."""
    body = client.get("/history").text
    for label in ("regions found", "clean", "failed to read"):
        assert label in body
