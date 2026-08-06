"""Material coverage and thresholds (T-06 / T-22).

Two properties are load-bearing: every coverage number on screen comes from the metrics
file, and every threshold comes from the one `Profile` definition in app/profiles.py,
which app/postprocess.py re-exports.  Both are checked by changing the source and
confirming the interface follows.
"""

from __future__ import annotations

import json

import pytest

from app.profiles import ACTIVE_PROFILE, Profile
from app.services import material_service


def test_every_material_is_listed(conn, settings):
    codes = {m["material_code"] for m in material_service.coverage(conn, settings)}
    assert codes == {"steel", "plastic", "ceramic", "epoxy", "glass", "non_steel_metal"}


def test_coverage_numbers_come_from_the_metrics_file(conn, settings):
    metrics = json.loads(settings.metrics_file.read_text())
    by_code = {m["material_code"]: m for m in metrics["materials"]}
    for row in material_service.coverage(conn, settings):
        expected = by_code[row["material_code"]]
        assert row["class_typing"] == expected["class_typing"]
        assert row["training_masks"] == expected["training_masks"]
        assert row["notes"] == expected["notes"]


def test_screen_shows_the_metrics_file_values(client, settings):
    body = client.get("/materials").text
    metrics = json.loads(settings.metrics_file.read_text())
    for material in metrics["materials"]:
        if material["class_typing"] is not None:
            assert str(material["class_typing"]) in body
        assert material["material_name"] in body


def test_limited_materials_are_marked_as_limited(conn, settings):
    rows = {m["material_code"]: m for m in material_service.coverage(conn, settings)}
    assert rows["steel"]["is_limited"] is False
    for code in ("plastic", "ceramic", "epoxy", "glass", "non_steel_metal"):
        assert rows[code]["is_limited"] is True


def test_unsupported_materials_are_labelled_on_screen(client):
    body = client.get("/materials").text
    for label in ("not supported", "one product only", "thin coverage", "typing unsupported"):
        assert label in body


def test_non_steel_metal_is_not_presented_as_supported(conn, settings):
    row = next(m for m in material_service.coverage(conn, settings) if m["material_code"] == "non_steel_metal")
    assert row["support_status"] == "not_supported"
    assert row["class_typing"] is None


def test_thresholds_come_from_the_module(conn):
    thresholds = material_service.thresholds(conn)
    assert thresholds["stored"]["crack_threshold"] == ACTIVE_PROFILE.crack_thresh
    assert thresholds["stored"]["scratch_threshold"] == ACTIVE_PROFILE.scratch_thresh
    assert thresholds["stored"]["minimum_area_px"] == ACTIVE_PROFILE.min_area_px
    assert thresholds["stored"]["minimum_skeleton_px"] == ACTIVE_PROFILE.min_skeleton_px
    assert thresholds["in_sync"] is True
    assert thresholds["source"].startswith("app/profiles.py::Profile")


def test_postprocess_reexports_the_same_profile_object():
    """One definition, two import paths (T-06).

    `profiles.py` is stdlib-only so the interface can read the thresholds where cv2 is
    not installed; `postprocess.py` re-exports it so the pipeline and the interface
    cannot end up holding different numbers. Skipped where the inference core's
    dependencies are absent, which is exactly the environment the split exists for.
    """
    postprocess = pytest.importorskip(
        "app.postprocess", reason="inference core dependencies (cv2/numpy/skimage) not installed"
    )
    from app import profiles

    assert postprocess.Profile is profiles.Profile
    assert postprocess.ACTIVE_PROFILE is profiles.ACTIVE_PROFILE


def test_retuning_the_module_flows_through_to_the_interface(monkeypatch, tmp_path, settings, client):
    """T-06: change Profile, re-seed, and both the database and the screen follow.

    No threshold value is edited anywhere else for this to work.
    """
    retuned = Profile(crack_thresh=0.55, scratch_thresh=0.31, min_area_px=40, min_skeleton_px=9)

    monkeypatch.setattr("app.database.seed.ACTIVE_PROFILE", retuned)
    monkeypatch.setattr("app.services.material_service.ACTIVE_PROFILE", retuned)
    monkeypatch.setattr("app.providers.mock_assets.ACTIVE_PROFILE", retuned)

    from app.database.connection import connect
    from app.database.seed import seed_reference_data

    scratch_db = tmp_path / "retuned.db"
    monkeypatch.setattr(type(settings), "db_file", property(lambda self: scratch_db))

    connection = connect(scratch_db)
    try:
        from app.database.connection import create_schema

        create_schema(connection)
        seed_reference_data(connection, settings)
        thresholds = material_service.thresholds(connection)
    finally:
        connection.close()

    assert thresholds["stored"]["crack_threshold"] == 0.55
    assert thresholds["stored"]["scratch_threshold"] == 0.31
    assert thresholds["stored"]["minimum_area_px"] == 40
    assert thresholds["stored"]["minimum_skeleton_px"] == 9


def test_threshold_api_returns_the_active_values(client):
    payload = client.get("/api/thresholds").json()
    assert payload["stored"]["crack_threshold"] == ACTIVE_PROFILE.crack_thresh
    assert payload["in_sync"] is True


def test_model_metadata_is_reported(conn, settings):
    model = material_service.model_info(conn, settings)
    assert model["file_name"]
    assert model["artefact_sha256"]
    assert model["parameter_count"] == 1430000
    assert model["size_mb"] == 5.8
    assert model["precision"] == "float32"
    assert model["latency_ms"] == 26.0
    assert "ARM" in model["arm_note"]


def test_materials_screen_states_the_typing_limitation(client):
    body = client.get("/materials").text.lower()
    assert "provisional" in body
    assert "transfer" in body


def test_materials_api_shape(client):
    payload = client.get("/api/materials").json()
    assert len(payload["materials"]) == 6
    assert payload["thresholds"]["in_sync"] is True
    assert payload["overall"]["detection_rate"] == 0.96
