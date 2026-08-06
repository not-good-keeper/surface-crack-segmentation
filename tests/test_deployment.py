"""Local-only deployment and portable-path guarantees."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_local_defaults_bind_loopback_and_use_real_provider():
    from app.config import Settings

    settings = Settings(_env_file=None, inspection_provider="real", demo_mode=False)
    assert settings.app_host == "127.0.0.1"
    assert settings.inspection_provider == "real"
    assert settings.demo_mode is False


def test_runtime_paths_stay_under_the_local_data_root():
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        data_root=Path("data"), database_path=Path("data/inspection.db"),
        source_root=Path("data/sources"), overlay_root=Path("data/overlays"),
        export_root=Path("data/exports"), batch_root=Path("data/batches"),
        log_root=Path("data/logs"),
    )
    for path in (settings.db_file, settings.source_dir, settings.overlay_dir,
                 settings.export_dir, settings.batch_dir, settings.log_dir):
        assert path.resolve().is_relative_to((ROOT / "data").resolve())


def test_no_vercel_entry_or_configuration_remains():
    assert not (ROOT / "vercel.json").exists()
    assert not (ROOT / ".vercelignore").exists()
    assert not (ROOT / "api" / "index.py").exists()


def test_runtime_requirements_include_local_cpu_inference():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for dependency in ("fastapi", "onnxruntime", "opencv-python-headless", "scikit-image"):
        assert dependency in requirements


def test_stored_image_paths_are_relative_to_data_root(settings):
    from app.services.inspection_service import portable_media_path

    absolute = settings.overlay_dir / "real" / "example.png"
    assert portable_media_path(str(absolute), settings) == "overlays/real/example.png"


def test_relative_traversal_in_a_stored_path_is_refused(settings):
    from app.services.inspection_service import image_path_for

    assert image_path_for({"source_image_path": "../../../etc/passwd"}, "source", settings) is None


def test_dockerfile_runs_mock_mode_only():
    """The container path (docs/DEPLOYMENT.md §0) must never run real inference.

    A public host and a factory image are incompatible by design (NFR-05); the image
    should not even have the inference stack installed for `real` mode to fall back to.
    """
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "INSPECTION_PROVIDER=mock" in text
    install_lines = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    ).lower()
    for package in ("onnxruntime", "opencv", "scikit-image"):
        assert package not in install_lines


def test_dockerfile_respects_the_hosting_platform_port_and_binds_publicly():
    """Render/Railway/Fly.io/Cloud Run all inject $PORT; the local default stays 8000."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "${PORT:-8000}" in text
    assert "--host 0.0.0.0" in text
    assert "HEALTHCHECK" in text


def test_dockerignore_excludes_the_training_corpus():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in ("data/", ".venv-app/", ".git/", "bench/", "dataset/"):
        assert entry in text


def test_render_blueprint_builds_the_dockerfile_in_mock_mode():
    """render.yaml is what makes `New + -> Blueprint` on Render need zero manual setup.

    Kept as a plain text check, not a YAML parse: no test dependency here should force
    PyYAML into requirements-dev.txt just to read six lines of config.
    """
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "runtime: docker" in text
    assert "dockerfilePath: ./Dockerfile" in text
    assert "plan: free" in text
    assert "healthCheckPath: /healthz" in text
    assert "value: mock" in text


def test_relative_paths_resolve_against_local_data_root(seeded, settings):
    from app.services.inspection_service import image_path_for, portable_media_path

    sample = next(settings.overlay_dir.rglob("*.png"), None)
    if sample is None:
        pytest.skip("no generated overlay available")
    relative = portable_media_path(str(sample), settings)
    assert not Path(relative).is_absolute()
    assert image_path_for({"overlay_image_path": relative}, "overlay", settings) == sample.resolve()
