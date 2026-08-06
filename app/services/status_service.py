"""Station health checks: camera, model file and hash, database, disk space.

The result of these checks is what the persistent health strip shows on every screen.
A failing check is never smoothed over: the Live screen switches to CHECK STATION and
inspection stops, because a system that cannot identify its own model file must not
keep producing results (NFR-13 / T-23).
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.config import Settings, get_settings
from app.database.migrations import verify_schema
from app.providers import provider_error
from app.repositories import inspection_repository as inspections
from app.repositories import model_repository as models

OK = "ok"
WARNING = "warning"
FAILED = "failed"

#: Demo-only fault injection.  Mock mode must be able to show a failing station as well
#: as a healthy one; this is the control behind that, and it is labelled as such in the
#: interface.  It has no effect when INSPECTION_PROVIDER=real.
SIMULATIONS = {
    "none": "All checks report their real state",
    "camera_offline": "Camera check reports offline",
    "model_hash_mismatch": "Model hash check reports a mismatch",
    "disk_low": "Disk check reports below the free-space threshold",
    "database_error": "Database check reports an error",
}

_simulation = "none"


def set_simulation(name: str) -> str:
    global _simulation
    _simulation = name if name in SIMULATIONS else "none"
    return _simulation


def get_simulation() -> str:
    return _simulation


def _check(
    key: str,
    label: str,
    state: str,
    detail: str,
    action: str = "",
    value: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "state": state,
        "state_label": {OK: "OK", WARNING: "DEGRADED", FAILED: "FAILED"}[state],
        "detail": detail,
        "action": action,
        "value": value,
        "checked_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
    }


def camera_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    if settings.is_mock and _simulation == "camera_offline":
        return _check(
            "camera", "Camera", FAILED,
            f"Simulated fault: {settings.station_id} is not delivering frames.",
            "Check the camera cable and power, then run the checks again.",
            "offline",
        )
    row = models.station_by_code(conn, settings.station_id)
    if row is None:
        return _check(
            "camera", "Camera", WARNING,
            f"Station {settings.station_id} is not registered in the database.",
            "Seed the database or correct STATION_ID.", "unknown",
        )
    if settings.is_mock:
        return _check(
            "camera", "Camera", OK,
            f"{settings.station_id} simulated by the mock provider - no physical camera is attached.",
            "", "simulated",
        )
    return _check(
        "camera", "Camera", OK if row["camera_status"] == "ok" else FAILED,
        f"{settings.station_id} reports {row['camera_status']}.",
        "" if row["camera_status"] == "ok" else "Inspect the camera at the station.",
        row["camera_status"],
    )


def model_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    """Verify the model file and its SHA-256.

    In mock mode a missing file is expected and reported as DEGRADED, not FAILED: the
    interface is explicitly designed to run before the weights exist.  In real mode a
    missing file or a hash mismatch is FAILED and blocks inspection.
    """
    row = models.active_model(conn) or {}
    expected = (settings.model_sha256 or row.get("artefact_sha256") or "").strip().lower()
    path = settings.model_file

    if settings.is_mock and _simulation == "model_hash_mismatch":
        return _check(
            "model", "Model file / hash", FAILED,
            "Simulated fault: the SHA-256 on disk does not match the configured hash. "
            "Inspection is stopped rather than producing results from an unknown model.",
            "Restore the configured model file or update MODEL_SHA256 deliberately.",
            "mismatch",
        )

    if not path.exists():
        if settings.is_mock:
            return _check(
                "model", "Model file / hash", WARNING,
                f"No ONNX file at {path}. Mock mode does not need one; results on screen "
                "are generated, not measured.",
                "Drop the exported model in place and set INSPECTION_PROVIDER=real.",
                "absent (mock mode)",
            )
        return _check(
            "model", "Model file / hash", FAILED,
            f"INSPECTION_PROVIDER=real but no model file at {path}.",
            "Set MODEL_PATH to the exported .onnx file.", "missing",
        )

    from app.providers.real_provider import sha256_of_file

    actual = sha256_of_file(path)
    if expected and actual.lower() != expected:
        return _check(
            "model", "Model file / hash", FAILED,
            f"Hash mismatch. Expected {expected[:12]}..., found {actual[:12]}....",
            "Inspection is stopped. Restore the configured file or record the new hash.",
            "mismatch",
        )
    err = provider_error()
    if err is not None:
        return _check("model", "Model file / hash", FAILED, err.message, "See MODEL_INTEGRATION.md.", err.code)
    return _check(
        "model", "Model file / hash", OK,
        f"{path.name} verified at load ({actual[:4]}\u2026{actual[-3:]}).", "", "verified",
    )


def database_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    if settings.is_mock and _simulation == "database_error":
        return _check(
            "database", "Database", FAILED, "Simulated fault: the database is not writable.",
            "Check permissions on the data folder.", "error",
        )
    try:
        missing = verify_schema(conn)
        if missing:
            return _check(
                "database", "Database", FAILED, f"Missing tables: {', '.join(missing)}.",
                "Run: python -m scripts.reset_db", "schema incomplete",
            )
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        count = inspections.total_count(conn)
        if integrity != "ok":
            return _check("database", "Database", FAILED, f"Integrity check: {integrity}.",
                          "Restore the database file from backup.", integrity)
        return _check(
            "database", "Database", OK,
            f"{settings.db_file.name}: 8 tables, {count:,} inspections stored.", "", "ok",
        )
    except sqlite3.Error as exc:
        return _check("database", "Database", FAILED, str(exc), "Check the database file.", "error")


def disk_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    if settings.is_mock and _simulation == "disk_low":
        return _check(
            "disk", "Disk space", FAILED,
            f"Simulated fault: free space is below the {settings.min_free_disk_gb} GB threshold. "
            "The run stops rather than dropping records silently.",
            "Archive older months and free space.", "below threshold",
        )
    try:
        usage = shutil.disk_usage(settings.db_file.parent)
    except OSError as exc:
        return _check("disk", "Disk space", WARNING, str(exc), "Check the data volume.", "unknown")

    free_gb = usage.free / 1024**3
    used_pct = round(usage.used / usage.total * 100)
    if free_gb < settings.min_free_disk_gb:
        return _check(
            "disk", "Disk space", FAILED,
            f"{free_gb:.1f} GB free, below the {settings.min_free_disk_gb} GB threshold.",
            "The run stops rather than dropping records. Archive older months.",
            f"{used_pct}% used",
        )
    return _check(
        "disk", "Disk space", OK, f"{free_gb:.1f} GB free ({used_pct}% used).", "", f"{used_pct}% used",
    )


def provider_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    err = provider_error()
    if err is not None:
        return _check("provider", "Inference provider", FAILED, err.message, "See MODEL_INTEGRATION.md.", err.code)
    if settings.is_mock:
        return _check(
            "provider", "Inference provider", WARNING,
            "Deterministic mock provider. Results are generated for demonstration and are "
            "not model measurements.",
            "Set INSPECTION_PROVIDER=real when the exported model is in place.", "mock",
        )
    return _check("provider", "Inference provider", OK, "Real provider active via app.inference.Inspector.", "", "real")


def last_inspection_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    row = inspections.latest(conn, settings.station_id) or inspections.latest(conn)
    if row is None:
        return _check(
            "last_inspection", "Last inspection", WARNING, "No inspection has been recorded yet.",
            "Run: python -m scripts.seed_db", "none",
        )
    return _check(
        "last_inspection", "Last inspection", OK,
        f"{row['product_id']} at {row['captured_at']} - {row['status'].replace('_', ' ')}.",
        "", str(row["inspection_id"]),
    )


def folder_check(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    problems = []
    for label, path in (
        ("overlay", settings.overlay_dir),
        ("source", settings.source_dir),
        ("batch", settings.batch_dir),
        ("export", settings.export_dir),
    ):
        if not path.exists():
            problems.append(f"{label} folder missing ({path})")
    if problems:
        return _check("folders", "Overlay / batch folders", WARNING, "; ".join(problems),
                      "Run the seed script to create them.", "incomplete")
    return _check("folders", "Overlay / batch folders", OK,
                  "Overlay, source, batch and export folders are present.", "", "ok")


CHECKS = (
    camera_check,
    model_check,
    database_check,
    disk_check,
    provider_check,
    last_inspection_check,
    folder_check,
)

PRIMARY_KEYS = ("camera", "model", "database", "disk")


def run_checks(conn: sqlite3.Connection, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    results = [check(conn, settings) for check in CHECKS]
    primary = [c for c in results if c["key"] in PRIMARY_KEYS]

    failed = [c for c in results if c["state"] == FAILED]
    degraded = [c for c in results if c["state"] == WARNING]
    overall = FAILED if failed else (WARNING if degraded else OK)

    return {
        "overall": overall,
        "overall_label": {OK: "STATION OK", WARNING: "DEGRADED", FAILED: "CHECK STATION"}[overall],
        "checked_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "checks": results,
        "primary": primary,
        "failing": [c["label"] for c in failed],
        "degraded": [c["label"] for c in degraded],
        "blocks_inspection": bool(failed),
        "simulation": _simulation,
        "simulations": SIMULATIONS,
        "provider": settings.inspection_provider,
    }


def health_strip(conn: sqlite3.Connection, settings: Settings | None = None) -> dict[str, Any]:
    """The compact strip repeated on every screen."""
    settings = settings or get_settings()
    status = run_checks(conn, settings)
    by_key = {c["key"]: c for c in status["checks"]}
    model = models.active_model(conn) or {}
    latest = inspections.latest(conn, settings.station_id) or inspections.latest(conn)
    sha = model.get("artefact_sha256") or ""

    from app.services.material_service import thresholds

    profile = thresholds(conn).get("stored", {})
    return {
        "station": settings.station_id,
        "camera": by_key.get("camera", {}),
        "model": by_key.get("model", {}),
        "database": by_key.get("database", {}),
        "disk": by_key.get("disk", {}),
        "model_version": model.get("version", "\u2014"),
        "model_size_mb": model.get("size_mb"),
        "model_sha_short": (sha[:4] + "\u2026") if sha else "\u2014",
        "profile_label": profile.get("material_code") or "all materials",
        "profile_version": profile.get("version_no"),
        "disk_value": by_key.get("disk", {}).get("value", ""),
        "latency_ms": (latest or {}).get("latency_ms"),
        "overall": status["overall"],
        "overall_label": status["overall_label"],
        "failing": status["failing"],
        "degraded": status["degraded"],
        "blocks_inspection": status["blocks_inspection"],
        "provider": settings.inspection_provider,
        "demo_mode": settings.demo_mode,
    }
