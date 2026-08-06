"""Station checks, fault simulation and model-hash validation (T-23)."""

from __future__ import annotations

import builtins
import hashlib

import pytest

from app.providers.base import ProviderUnavailable
from app.services import status_service


def test_all_four_primary_checks_are_present(conn, settings):
    status = status_service.run_checks(conn, settings)
    keys = {c["key"] for c in status["checks"]}
    for expected in ("camera", "model", "database", "disk"):
        assert expected in keys


def test_each_check_reports_state_detail_and_time(conn, settings):
    for check in status_service.run_checks(conn, settings)["checks"]:
        assert check["state"] in ("ok", "warning", "failed")
        assert check["state_label"]
        assert check["detail"]
        assert check["checked_at"]


def test_mock_mode_reports_a_missing_model_as_degraded_not_failed(conn, settings, reset_simulation):
    """Mock mode is designed to run before the weights exist; that is not a failure."""
    status = status_service.run_checks(conn, settings)
    model = next(c for c in status["checks"] if c["key"] == "model")
    assert model["state"] == "warning"
    assert status["blocks_inspection"] is False


def test_database_check_passes_on_a_seeded_database(conn, settings, reset_simulation):
    check = status_service.database_check(conn, settings)
    assert check["state"] == "ok"
    assert "inspections stored" in check["detail"]


@pytest.mark.parametrize(
    "simulation,key",
    [
        ("camera_offline", "camera"),
        ("model_hash_mismatch", "model"),
        ("disk_low", "disk"),
        ("database_error", "database"),
    ],
)
def test_each_simulated_fault_fails_its_check(conn, settings, simulation, key, reset_simulation):
    status_service.set_simulation(simulation)
    status = status_service.run_checks(conn, settings)
    check = next(c for c in status["checks"] if c["key"] == key)
    assert check["state"] == "failed"
    assert status["overall"] == "failed"
    assert status["blocks_inspection"] is True
    assert status["overall_label"] == "CHECK STATION"


def test_a_failing_check_reaches_the_health_strip(conn, settings, reset_simulation):
    status_service.set_simulation("camera_offline")
    strip = status_service.health_strip(conn, settings)
    assert strip["overall"] == "failed"
    assert "Camera" in strip["failing"]


def test_a_failing_check_stops_the_live_screen_showing_a_normal_result(client, reset_simulation):
    client.post("/api/status/check?simulate=camera_offline")
    payload = client.get("/api/live").json()
    assert payload["station_ok"] is False

    body = client.get("/live").text
    assert "CHECK STATION" in body

    blocked = client.post("/api/demo/next")
    assert blocked.status_code == 409

    client.post("/api/status/check?simulate=none")
    assert client.get("/api/live").json()["station_ok"] is True


def test_status_page_shows_every_check(client, reset_simulation):
    body = client.get("/status").text
    for label in ("Camera", "Model file / hash", "Database", "Disk space", "Inference provider"):
        assert label in body
    assert "Run checks again" in body


def test_rerunning_checks_updates_the_timestamp(client, reset_simulation):
    first = client.get("/api/status").json()["checked_at"]
    second = client.post("/api/status/check").json()["checked_at"]
    assert second >= first


def test_an_unknown_simulation_is_refused(client, reset_simulation):
    assert client.post("/api/status/check?simulate=nonsense").status_code == 400


# -- model hash validation (T-23) --------------------------------------------
def test_real_provider_refuses_a_missing_model(settings, monkeypatch):
    monkeypatch.setattr(settings, "inspection_provider", "real")
    from app.providers.real_provider import RealInspectionProvider

    with pytest.raises(ProviderUnavailable) as excinfo:
        RealInspectionProvider(settings)
    assert excinfo.value.code == "model_file_missing"


def test_real_provider_refuses_a_hash_mismatch(settings, monkeypatch, tmp_path):
    """Replacing the .onnx with a different file of the same name must stop the run."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"a different model entirely")
    monkeypatch.setattr(type(settings), "model_file", property(lambda self: model))
    monkeypatch.setattr(settings, "model_sha256", hashlib.sha256(b"the expected model").hexdigest())
    monkeypatch.setattr(settings, "inspection_provider", "real")

    from app.providers.real_provider import RealInspectionProvider

    with pytest.raises(ProviderUnavailable) as excinfo:
        RealInspectionProvider(settings)
    assert excinfo.value.code == "model_hash_mismatch"
    assert "mismatch" in str(excinfo.value).lower()


def test_real_provider_reports_a_missing_inference_core(settings, monkeypatch, tmp_path):
    """With app/inference.py absent, real mode must say so rather than start empty.

    The inference core now ships in this repository, so the import is forced to fail
    here. Before the merge this case arose on its own; keeping it means the message an
    operator would see if the core were ever removed from a deployment stays covered.
    """
    model = tmp_path / "model.onnx"
    model.write_bytes(b"weights")
    monkeypatch.setattr(type(settings), "model_file", property(lambda self: model))
    monkeypatch.setattr(settings, "model_sha256", hashlib.sha256(b"weights").hexdigest())

    real_import = builtins.__import__

    def no_inference_core(name, *args, **kwargs):
        if name == "app.inference":
            raise ImportError("No module named 'app.inference'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_inference_core)

    from app.providers.real_provider import RealInspectionProvider

    with pytest.raises(ProviderUnavailable) as excinfo:
        RealInspectionProvider(settings)
    assert excinfo.value.code == "inference_core_missing"


def test_real_provider_rejects_a_file_that_is_not_a_model(settings, monkeypatch, tmp_path):
    """The inference core is present, so a bogus .onnx fails when Inspector loads it.

    This is the case that replaced the missing-core one after the merge: the file
    passes the existence and hash checks and is still not a graph onnxruntime can
    open. It must stop the run, not yield an inspector that returns nothing.
    """
    pytest.importorskip("onnxruntime", reason="inference core dependencies not installed")

    model = tmp_path / "model.onnx"
    model.write_bytes(b"not an onnx graph")
    monkeypatch.setattr(type(settings), "model_file", property(lambda self: model))
    monkeypatch.setattr(settings, "model_sha256", hashlib.sha256(b"not an onnx graph").hexdigest())

    from app.providers.real_provider import RealInspectionProvider

    with pytest.raises(ProviderUnavailable) as excinfo:
        RealInspectionProvider(settings)
    assert excinfo.value.code == "inspector_init_failed"


def test_status_screen_reports_a_hash_mismatch_and_blocks(conn, settings, monkeypatch, tmp_path, reset_simulation):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"swapped")
    monkeypatch.setattr(type(settings), "model_file", property(lambda self: model))
    monkeypatch.setattr(settings, "model_sha256", "0" * 64)

    check = status_service.model_check(conn, settings)
    assert check["state"] == "failed"
    assert "mismatch" in check["detail"].lower()


def test_mock_provider_is_flagged_as_not_a_measurement(conn, settings, reset_simulation):
    check = status_service.provider_check(conn, settings)
    assert check["state"] == "warning"
    assert "not model measurements" in check["detail"]
