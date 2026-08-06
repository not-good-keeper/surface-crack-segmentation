"""Test fixtures.

Every test runs against a temporary database, temporary image folders and a temporary
batch root, so a test run never touches the working data/ directory and never leaves
anything behind.  The seed is small (16 live inspections, two short batch folders)
because the full demo seed generates several hundred images and takes minutes; the
sequence is the same one the full seed produces, just shorter.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

ANCHOR = datetime(2026, 8, 6, 11, 42, 7)


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("vision404")


@pytest.fixture(scope="session", autouse=True)
def configure_environment(workspace: Path):
    """Point the application at the temporary workspace before anything imports it."""
    os.environ["VISION404_ENV_FILE"] = str(workspace / "absent.env")  # ignore the repo .env
    os.environ["DATABASE_PATH"] = str(workspace / "test.db")
    os.environ["BATCH_ROOT"] = str(workspace / "batches")
    os.environ["OVERLAY_ROOT"] = str(workspace / "overlays")
    os.environ["SOURCE_ROOT"] = str(workspace / "sources")
    os.environ["EXPORT_ROOT"] = str(workspace / "exports")
    os.environ["INSPECTION_PROVIDER"] = "mock"
    os.environ["DEMO_MODE"] = "true"
    os.environ["MOCK_SEED"] = "404"
    os.environ["MODEL_PATH"] = str(workspace / "export" / "model.onnx")
    os.environ["METRICS_PATH"] = str(Path(__file__).resolve().parent.parent / "data" / "metrics" / "coverage.json")

    from app.config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings()
    settings.ensure_dirs()
    yield settings


@pytest.fixture(scope="session")
def seeded(configure_environment):
    """A seeded database shared by the whole session."""
    from app.database import seed as seed_module

    settings = configure_environment
    # Two short folders keep the suite quick while still covering multi-run reporting.
    seed_module.BATCH_FOLDERS = [
        ("2026-08-12-steel-lineB", "steel", 5),
        ("2026-08-11-ceramic-lineA", "ceramic", 4),
    ]
    summary = seed_module.seed(settings, anchor=ANCHOR, reset=True, live_count=16)
    return summary


@pytest.fixture()
def settings(configure_environment):
    return configure_environment


@pytest.fixture()
def conn(seeded, settings):
    from app.database.connection import connect

    connection = connect(settings.db_file)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def client(seeded):
    """FastAPI test client.

    The app's lifespan runs, which also proves the start-up path works: create the
    schema, and seed only when the database is empty.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def reset_simulation():
    from app.services import status_service

    status_service.set_simulation("none")
    yield
    status_service.set_simulation("none")


@pytest.fixture()
def temp_batch_folder(settings):
    """An empty folder inside the batch root, cleaned up afterwards."""
    folder = settings.batch_dir / "pytest-folder"
    folder.mkdir(parents=True, exist_ok=True)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)
