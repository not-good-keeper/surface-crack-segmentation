"""Typed application configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class Settings(BaseSettings):
    """Application settings.

    Defaults are the mock-mode defaults: the application starts and every screen works
    with no ONNX file present.
    """

    model_config = SettingsConfigDict(
        env_file=os.environ.get("VISION404_ENV_FILE", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_env: str = "development"

    demo_mode: bool = True
    inspection_provider: str = Field(default="mock")

    data_root: Path = Path("data")

    database_path: Path = Path("data/inspection.db")
    model_path: Path = Path("data/export/model.onnx")
    station_id: str = "line-1-cam-A"

    batch_root: Path = Path("data/batches")
    overlay_root: Path = Path("data/overlays")
    source_root: Path = Path("data/sources")
    export_root: Path = Path("data/exports")
    log_root: Path = Path("data/logs")
    metrics_path: Path = Path("data/metrics/coverage.json")

    #: Largest capture the interface accepts, in MB.
    max_upload_mb: int = 20

    min_free_disk_gb: float = 5.0

    #: SHA-256 the real provider must find on MODEL_PATH.  Empty in mock mode.
    model_sha256: str = ""

    #: Deterministic seed for the mock provider and the database seed.
    mock_seed: int = 404

    #: Live screen poll interval, milliseconds (Phase 2 spec: refresh every 2-3 s).
    live_poll_ms: int = 2500

    #: Caps the longest edge of generated demo images. 0 leaves them at full size.
    #: The deployed demo bundle uses a cap to keep the upload small; a station does not.
    mock_image_max_width: int = 0

    #: Run batch work on the request thread instead of a worker thread.
    batch_synchronous: bool = False

    @field_validator("inspection_provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"mock", "real"}:
            raise ValueError("INSPECTION_PROVIDER must be 'mock' or 'real'")
        return v

    # -- resolved absolute paths ------------------------------------------------
    # Reads and writes share one root: the data/ folder beside the code. The station
    # and the container both run on a normal writable filesystem, so the split that
    # the retired serverless path needed is gone.
    #
    # Stored image paths are recorded relative to that root, so a database built on
    # one machine still resolves on another.

    @property
    def bundled_data_dir(self) -> Path:
        return _abs(self.data_root)

    @property
    def runtime_data_dir(self) -> Path:
        return self.bundled_data_dir

    def _runtime(self, configured: Path) -> Path:
        """Resolve a configured folder against the writable root."""
        path = Path(configured)
        if path.is_absolute():
            return path
        parts = path.parts
        if parts and parts[0] == self.data_root.name:
            return self.runtime_data_dir.joinpath(*parts[1:])
        return PROJECT_ROOT / path

    def _bundled(self, configured: Path) -> Path:
        """Resolve a configured folder against the read-only shipped root."""
        path = Path(configured)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def db_file(self) -> Path:
        return self._runtime(self.database_path)

    @property
    def bundled_db_file(self) -> Path:
        return self._bundled(self.database_path)

    @property
    def model_file(self) -> Path:
        return self._bundled(self.model_path)

    @property
    def batch_dir(self) -> Path:
        return self._runtime(self.batch_root)

    @property
    def overlay_dir(self) -> Path:
        return self._runtime(self.overlay_root)

    @property
    def source_dir(self) -> Path:
        return self._runtime(self.source_root)

    @property
    def export_dir(self) -> Path:
        return self._runtime(self.export_root)

    @property
    def log_dir(self) -> Path:
        return self._runtime(self.log_root)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def metrics_file(self) -> Path:
        return self._bundled(self.metrics_path)

    @property
    def media_roots(self) -> list[Path]:
        """Every folder a stored image is allowed to live in.

        Used to check a path that came out of the database before it is read - the
        database is trusted, but not blindly.
        """
        roots = [
            self.source_dir,
            self.overlay_dir,
            self.batch_dir,
            self.runtime_data_dir,
            self.bundled_data_dir,
        ]
        seen: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.append(resolved)
        return seen

    @property
    def is_mock(self) -> bool:
        return self.inspection_provider == "mock"

    @property
    def run_batches_synchronously(self) -> bool:
        return self.batch_synchronous

    def ensure_dirs(self) -> None:
        for d in (
            self.db_file.parent,
            self.batch_dir,
            self.overlay_dir,
            self.source_dir,
            self.export_dir,
            self.metrics_file.parent,
            self.model_file.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that point the application at a temporary database."""
    get_settings.cache_clear()
