"""Deterministic demo seed.

Produces a database that exercises every state the interface has to render: clean
parts, crack-only, scratch-only and mixed results, acquisition failures, processing
failures, several materials, several stations, several days, and three batch runs.

Determinism
-----------
Statuses, region geometry and image pixels are a pure function of ``mock_seed``.
Timestamps are laid out backwards from an anchor date, which defaults to today so the
history screen looks current; pass ``--anchor 2026-08-06`` to pin it for a byte-exact
reproduction.  ``tests/test_seed.py`` seeds twice with a fixed anchor and compares.

The broken files planted in the batch folders are genuinely broken - a zero-byte file,
a truncated PNG and a text file wearing a .png extension - so the failure path is
exercised by real decode failures rather than by a simulated flag (T-08).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.database.connection import connect, create_schema
from app.profiles import ACTIVE_PROFILE
from app.providers.mock_provider import MockInspectionProvider
from app.repositories import material_repository as materials
from app.services import batch_service, inspection_service
from app.services.material_service import load_metrics

LIVE_INSPECTIONS = 90

STATIONS = [
    ("line-1-cam-A", "line-1", 0.0182),
    ("line-1-cam-B", "line-1", 0.0182),
    ("line-2-cam-A", "line-2", None),  # not calibrated yet
]

DEFECT_CLASSES = [(1, "crack", "Crack"), (2, "scratch", "Scratch")]

#: Guaranteed coverage: the first entries force one of each state so a freshly seeded
#: database always contains every case the screens must handle.
FORCED_SEQUENCE = [
    "mixed", "clean", "crack_only", "acquisition_failure", "scratch_only",
    "processing_failure", "mixed", "clean", "crack_only", "scratch_only",
    "acquisition_failure", "processing_failure",
]

BATCH_FOLDERS = [
    ("2026-08-12-steel-lineB", "steel", 18),
    ("2026-08-11-ceramic-lineA", "ceramic", 14),
    ("2026-08-10-plastic-lineB", "plastic", 12),
]

TRUNCATED_PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000001000000010008060000")


def seed_reference_data(conn: sqlite3.Connection, settings: Settings) -> None:
    """Materials, classes, model version, stations and threshold profiles."""
    metrics = load_metrics(settings)

    for entry in metrics.get("materials", []):
        conn.execute(
            "INSERT OR IGNORE INTO material (material_code, material_name, support_status, notes) "
            "VALUES (?, ?, ?, ?)",
            [entry["material_code"], entry["material_name"], entry["support_status"], entry.get("notes")],
        )

    for class_id, code, name in DEFECT_CLASSES:
        conn.execute(
            "INSERT OR IGNORE INTO defect_class (class_id, class_code, display_name) VALUES (?, ?, ?)",
            [class_id, code, name],
        )

    model = metrics.get("model", {})
    conn.execute(
        """
        INSERT OR IGNORE INTO model_version
            (file_name, version, artefact_sha256, parameter_count, size_mb, precision,
             latency_ms, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        [
            model.get("file_name", "model.onnx"),
            model.get("version", "v8a"),
            model.get("artefact_sha256", "0" * 64),
            model.get("parameter_count"),
            model.get("size_mb"),
            model.get("precision"),
            model.get("latency_ms"),
            metrics.get("generated_at", datetime.now().isoformat(timespec="seconds")),
        ],
    )

    now = datetime.now().isoformat(timespec="seconds")
    codes = [s[0] for s in STATIONS]
    if settings.station_id not in codes:
        STATIONS.append((settings.station_id, settings.station_id.split("-cam")[0], None))
    for code, line, mm_per_pixel in STATIONS:
        conn.execute(
            "INSERT OR IGNORE INTO station (station_code, line_code, mm_per_pixel, camera_status, created_at) "
            "VALUES (?, ?, ?, 'ok', ?)",
            [code, line, mm_per_pixel, now],
        )

    # Threshold profiles mirror app/postprocess.py::Profile - read from the module, never
    # typed in.  A global profile plus a steel profile with the same values, which is
    # what the Phase 2 wireframe shows as "Profile steel v2".
    p = ACTIVE_PROFILE
    conn.execute(
        """
        INSERT OR IGNORE INTO profile
            (material_id, version_no, crack_threshold, scratch_threshold,
             minimum_area_px, minimum_skeleton_px, created_at, is_active)
        VALUES (NULL, ?, ?, ?, ?, ?, ?, 1)
        """,
        [p.version_no, p.crack_thresh, p.scratch_thresh, p.min_area_px, p.min_skeleton_px, now],
    )
    steel_id = materials.material_id(conn, "steel")
    if steel_id:
        conn.execute(
            """
            INSERT OR IGNORE INTO profile
                (material_id, version_no, crack_threshold, scratch_threshold,
                 minimum_area_px, minimum_skeleton_px, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            [steel_id, p.version_no, p.crack_thresh, p.scratch_thresh, p.min_area_px, p.min_skeleton_px, now],
        )
    conn.commit()


def seed_live_inspections(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: MockInspectionProvider,
    anchor: datetime,
    count: int = LIVE_INSPECTIONS,
) -> int:
    """Inspections spread over recent days, stations and materials."""
    written = 0
    # Counted down so the newest part is written last and is therefore the live one.
    for index in reversed(range(count)):
        key = f"live-{index:05d}"
        force = FORCED_SEQUENCE[index] if index < len(FORCED_SEQUENCE) else None
        material = provider.material_for(key)
        station = STATIONS[index % len(STATIONS)][0]
        # Newest first: index 0 is the most recent part at the station.
        captured = anchor - timedelta(
            days=index // 8, minutes=(index % 8) * 17 + 3, seconds=(index * 7) % 59
        )
        product_id = f"batch-{77 - index // 24}/item-{index % 24 + 1:02d}"

        result = provider.inspect(
            None, None,
            product_id=product_id,
            material=material,
            key=key,
            force_kind=force,
            captured_at=captured,
            station_id=station,
        )
        inspection_service.store_result(conn, result, settings=settings, write_log=False)
        written += 1
    conn.commit()
    return written


def build_batch_folders(settings: Settings, provider: MockInspectionProvider) -> list[tuple[str, str]]:
    """Write real image files (and three real broken files) into the batch root."""
    created = []
    for folder_name, material, image_count in BATCH_FOLDERS:
        folder = settings.batch_dir / folder_name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(image_count):
            name = f"IMG_{3110 + i:04d}.png"
            key = f"batch:{name}"
            result = provider.inspect(
                None, None, product_id=name, material=material, key=key, station_id=settings.station_id
            )
            if result.source_image_path:
                shutil.copyfile(result.source_image_path, folder / name)

        # Genuinely unreadable files - decode failures, not simulated ones.
        (folder / f"IMG_{3110 + image_count:04d}.png").write_bytes(b"")
        (folder / f"IMG_{3111 + image_count:04d}.png").write_bytes(TRUNCATED_PNG)
        (folder / f"IMG_{3112 + image_count:04d}.png").write_text(
            "this is a text file wearing a .png extension\n", encoding="utf-8"
        )
        created.append((folder_name, material))
    return created


def seed_batch_runs(settings: Settings, provider: MockInspectionProvider) -> list[int]:
    folders = build_batch_folders(settings, provider)
    run_ids = []
    for folder_name, material in folders:
        info = batch_service.start_run(
            folder_name,
            material,
            product_prefix=f"{folder_name[:10]}/",
            settings=settings,
            synchronous=True,
        )
        run_ids.append(info["batch_run_id"])
    return run_ids


def seed(
    settings: Settings | None = None,
    *,
    anchor: datetime | None = None,
    reset: bool = False,
    live_count: int = LIVE_INSPECTIONS,
    with_batches: bool = True,
) -> dict[str, Any]:
    settings = settings or get_settings()
    settings.ensure_dirs()

    if reset:
        for path in (settings.db_file, Path(str(settings.db_file) + "-wal"), Path(str(settings.db_file) + "-shm")):
            path.unlink(missing_ok=True)
        for folder in (settings.source_dir / "mock", settings.overlay_dir / "mock"):
            if folder.exists():
                shutil.rmtree(folder)
        log = settings.export_dir.parent / inspection_service.RECORD_LOG_NAME
        log.unlink(missing_ok=True)

    anchor = anchor or datetime.now().replace(microsecond=0)
    provider = MockInspectionProvider(settings)

    conn = connect(settings.db_file)
    try:
        create_schema(conn)
        seed_reference_data(conn, settings)
    finally:
        conn.close()

    # Batches first, live inspections last: the Live screen shows the most recently
    # processed part, and after seeding that should be a capture from the station
    # rather than the last file of a batch folder.
    run_ids = seed_batch_runs(settings, provider) if with_batches else []

    conn = connect(settings.db_file)
    try:
        live = seed_live_inspections(conn, settings, provider, anchor, live_count)
    finally:
        conn.close()

    conn = connect(settings.db_file)
    try:
        totals = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM inspection GROUP BY status")
        }
        total = conn.execute("SELECT COUNT(*) AS n FROM inspection").fetchone()["n"]
        regions = conn.execute("SELECT COUNT(*) AS n FROM defect_region").fetchone()["n"]
    finally:
        conn.close()

    return {
        "live_inspections": live,
        "batch_runs": run_ids,
        "total_inspections": total,
        "total_regions": regions,
        "status_totals": totals,
        "database": str(settings.db_file),
        "anchor": anchor.isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the inspection database with deterministic demo data.")
    parser.add_argument("--reset", action="store_true", help="delete the database and generated images first")
    parser.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for reproducible timestamps")
    parser.add_argument("--live", type=int, default=LIVE_INSPECTIONS, help="number of live inspections")
    parser.add_argument("--no-batches", action="store_true", help="skip the batch runs")
    args = parser.parse_args()

    anchor = datetime.fromisoformat(args.anchor) if args.anchor else None
    summary = seed(
        anchor=anchor, reset=args.reset, live_count=args.live, with_batches=not args.no_batches
    )

    print(f"Database:          {summary['database']}")
    print(f"Anchor:            {summary['anchor']}")
    print(f"Live inspections:  {summary['live_inspections']}")
    print(f"Batch runs:        {summary['batch_runs']}")
    print(f"Total inspections: {summary['total_inspections']}")
    print(f"Total regions:     {summary['total_regions']}")
    for status, n in sorted(summary["status_totals"].items()):
        print(f"  {status:<22} {n}")


if __name__ == "__main__":
    main()
