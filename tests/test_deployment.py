"""Deployment configuration: portable paths and read-only/serverless behaviour."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def serverless_settings(monkeypatch, tmp_path):
    from app.config import Settings

    monkeypatch.setenv("VERCEL", "1")
    settings = Settings(
        _env_file=None,
        database_path=Path("data/inspection.db"),
        batch_root=Path("data/batches"),
        overlay_root=Path("data/overlays"),
        source_root=Path("data/sources"),
        export_root=Path("data/exports"),
        model_path=Path("data/export/model.onnx"),
        metrics_path=Path("data/metrics/coverage.json"),
        runtime_root=tmp_path / "runtime",
    )
    return settings


# -- configuration -------------------------------------------------------------
def test_serverless_writes_go_to_the_runtime_root(serverless_settings, tmp_path):
    assert serverless_settings.is_serverless is True
    assert serverless_settings.db_file == tmp_path / "runtime" / "inspection.db"
    assert serverless_settings.overlay_dir == tmp_path / "runtime" / "overlays"


def test_serverless_reads_metrics_and_model_from_the_bundle(serverless_settings):
    """Read-only files stay in the deployment; only writes are redirected."""
    assert serverless_settings.metrics_file.is_relative_to(ROOT)
    assert serverless_settings.model_file.is_relative_to(ROOT)
    assert serverless_settings.bundled_db_file.is_relative_to(ROOT)


def test_serverless_runs_batches_synchronously(serverless_settings):
    """A worker thread is frozen once the response is sent, so the work must not use one."""
    assert serverless_settings.run_batches_synchronously is True


def test_local_settings_keep_one_data_root(settings):
    assert settings.run_batches_synchronously is False


def test_batch_roots_include_the_bundled_folders_when_read_only(serverless_settings):
    from app.services import batch_service

    roots = [str(r) for r in batch_service.batch_roots(serverless_settings)]
    assert any("runtime" in r for r in roots)
    assert any(str(ROOT / "data" / "batches") == r for r in roots)


# -- portable media paths -------------------------------------------------------
def test_stored_image_paths_are_relative_to_a_data_root(serverless_settings, tmp_path):
    from app.services.inspection_service import portable_media_path

    absolute = str(ROOT / "data" / "overlays" / "mock" / "example_ovl.png")
    assert portable_media_path(absolute, serverless_settings) == "overlays/mock/example_ovl.png"


def test_a_path_outside_every_root_is_left_absolute(serverless_settings, tmp_path):
    """A path that belongs to no configured root is passed through, not made relative.

    The path is built from tmp_path rather than written as "/etc/passwd": on Windows
    that string resolves to D:\\etc\\passwd, so the literal was asserting the platform
    rather than the behaviour.
    """
    from app.services.inspection_service import portable_media_path

    outsider = (tmp_path / "elsewhere" / "capture.png").resolve()
    assert portable_media_path(str(outsider), serverless_settings) == str(outsider)


def test_relative_paths_resolve_against_the_data_root(seeded, settings):
    """A database built on one machine still finds its images on another.

    The stored path is relative to the data root, so moving the folder - or reading it
    from a different filesystem after a deployment - still resolves.
    """
    from app.config import Settings
    from app.services.inspection_service import image_path_for, portable_media_path

    sample = next((settings.overlay_dir).rglob("*.png"), None)
    if sample is None:
        pytest.skip("no generated overlay available")

    # A configuration whose data root is the folder those images actually live in.
    portable = Settings(
        _env_file=None,
        runtime_root=settings.overlay_dir.parent,
        overlay_root=Path("data/overlays"),
        source_root=Path("data/sources"),
        batch_root=Path("data/batches"),
    )
    relative = portable_media_path(str(sample), portable)
    assert not Path(relative).is_absolute()
    assert relative.startswith("overlays/")

    resolved = image_path_for({"overlay_image_path": relative}, "overlay", portable)
    assert resolved == sample.resolve()


def test_relative_traversal_in_a_stored_path_is_still_refused(settings):
    from app.services.inspection_service import image_path_for

    assert image_path_for({"source_image_path": "../../../etc/passwd"}, "source", settings) is None


# -- deployment files -----------------------------------------------------------
def test_vercel_config_is_valid_and_routes_everything_to_the_app():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert config["builds"][0]["src"] == "api/index.py"
    assert config["routes"][-1]["dest"] == "api/index.py"
    assert config["env"]["INSPECTION_PROVIDER"] == "mock"
    assert config["env"]["SERVERLESS"] == "true"


def test_vercel_bundles_the_files_that_are_not_python_imports():
    """Templates, static assets and the demo bundle must be declared explicitly.

    The Python builder traces imports. Jinja templates, CSS, JS and the seeded database
    are opened by path at runtime, so nothing traces them and they are silently left out
    of the function — the deployment builds, and then every page 500s on a missing
    template. `includeFiles` is the only thing that puts them in.
    """
    config = json.loads((ROOT / "vercel.json").read_text())
    included = config["builds"][0]["config"]["includeFiles"]
    for needed in ("app/templates", "app/static", "data/inspection.db", "data/metrics"):
        assert needed in included, f"{needed} would not be deployed"


def test_vercel_entry_point_exposes_an_asgi_app():
    text = (ROOT / "api" / "index.py").read_text()
    assert "from app.main import app" in text


def test_runtime_requirements_do_not_pull_in_the_inference_stack():
    """Mock mode must install without onnxruntime; real mode opts in deliberately."""
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = " ".join(lines).lower()
    assert "fastapi" in joined
    assert "onnxruntime" not in joined
    assert "opencv" not in joined


def test_mock_mode_imports_nothing_from_the_inference_stack():
    """The declaration in requirements.txt has to be true of the import graph too.

    The pipeline now lives in this repository, so `app.postprocess` and `app.inference`
    are one import away from everything. If any module on the mock path reached for
    them at import time, mock mode would need cv2, numpy's full stack, scikit-image and
    onnxruntime installed — roughly 290 MB, over the serverless function limit — and
    the failure would only appear at deploy time.

    Run in a subprocess with those modules poisoned, because the test session itself has
    them installed and would otherwise import them successfully.
    """
    import subprocess
    import sys

    probe = """
import sys

class Blocked:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"cv2", "onnxruntime", "skimage", "torch"}:
            raise ImportError("blocked by the deployment probe: " + name)
        return None

sys.meta_path.insert(0, Blocked())
import app.main
import app.providers.mock_provider
import app.services.material_service
import app.database.seed
assert "cv2" not in sys.modules
assert "onnxruntime" not in sys.modules
print("clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "INSPECTION_PROVIDER": "mock"},
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_deployed_bundle_meets_the_seed_floor():
    """The shipped demo database must still carry at least 75 inspections."""
    import sqlite3

    database = ROOT / "data" / "inspection.db"
    if not database.exists():
        pytest.skip("no demo bundle built")
    connection = sqlite3.connect(database)
    try:
        total = connection.execute("SELECT COUNT(*) FROM inspection").fetchone()[0]
        statuses = {
            row[0] for row in connection.execute("SELECT DISTINCT status FROM inspection")
        }
        batches = connection.execute("SELECT COUNT(*) FROM batch_run").fetchone()[0]
    finally:
        connection.close()

    assert total >= 75
    assert batches >= 3
    assert statuses == {"regions_found", "clean", "acquisition_failure", "processing_failure"}
