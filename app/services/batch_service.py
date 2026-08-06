"""Batch runs over a folder.

Two things matter here.

Safety: the source folder is never taken as a raw filesystem path.  A request names a
folder *inside* the configured BATCH_ROOT; the name is resolved and then checked to be
inside that root, so ``../../etc`` cannot escape (Phase 2 report 3.11).

Reconciliation: the summary cards, the results table and both exports are all answered
from the same query over the inspection rows the run produced.  Nothing is counted
twice and a file that could not be read is never counted as clean.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.database.connection import connect
from app.providers import get_provider
from app.repositories import batch_repository as batches
from app.repositories import inspection_repository as inspections
from app.repositories import material_repository as materials
from app.repositories.inspection_repository import HistoryFilters
from app.services import inspection_service

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_FILE_BYTES = 64 * 1024 * 1024

_progress: dict[int, dict[str, Any]] = {}
_progress_lock = threading.Lock()


class UnsafeFolder(ValueError):
    """Raised when a requested folder resolves outside the configured batch root."""


# -- folder handling ------------------------------------------------------------
def batch_roots(settings: Settings | None = None) -> list[Path]:
    """Every folder a batch may be read from.

    Normally one folder. On a read-only deployment there are two: the writable root
    under /tmp, and the folders shipped with the build. Both are configured locations,
    so both are safe; anything outside them is not.
    """
    settings = settings or get_settings()
    roots = [settings.batch_dir]
    bundled = settings._bundled(settings.batch_root)
    if bundled.resolve() != settings.batch_dir.resolve():
        roots.append(bundled)
    return [r for r in roots if r.exists() or r == settings.batch_dir]


def list_folders(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    folders: dict[str, dict[str, Any]] = {}
    for root in batch_roots(settings):
        root.mkdir(parents=True, exist_ok=True)
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            # A writable folder of the same name shadows a shipped one.
            if child.name in folders:
                continue
            folders[child.name] = {
                "name": child.name,
                "path": str(child),
                "relative": child.name,
                "image_count": len(image_files(child)),
            }
    return list(folders.values())


def resolve_folder(name: str, settings: Settings | None = None) -> Path:
    """Resolve a requested folder inside a configured batch root, or refuse it."""
    settings = settings or get_settings()
    raw = (name or "").strip()
    if not raw:
        raise UnsafeFolder("No source folder was given.")

    roots = [r.resolve() for r in batch_roots(settings)]
    candidate = Path(raw)

    for root in roots:
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not target.is_relative_to(root):
            continue
        if target.exists() and target.is_dir():
            return target

    inside_a_root = any(
        (candidate if candidate.is_absolute() else root / candidate).resolve().is_relative_to(root)
        for root in roots
    )
    if not inside_a_root:
        raise UnsafeFolder(
            f"'{raw}' resolves outside the configured batch root "
            f"({', '.join(str(r) for r in roots)}). Choose one of the configured folders."
        )
    raise UnsafeFolder(f"'{raw}' is not a folder inside the batch root.")


def image_files(folder: Path) -> list[Path]:
    """Candidate files, extension-checked and size-limited.

    A file whose extension passes but whose content is not an image still reaches the
    provider, decodes there and is recorded as an acquisition failure - which is the
    behaviour T-08 asks for.
    """
    out = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(path)
    return out


# -- progress -------------------------------------------------------------------
def get_progress(batch_run_id: int) -> dict[str, Any] | None:
    with _progress_lock:
        state = _progress.get(batch_run_id)
        return dict(state) if state else None


def _set_progress(batch_run_id: int, **fields: Any) -> None:
    with _progress_lock:
        state = _progress.setdefault(batch_run_id, {})
        state.update(fields)


# -- running --------------------------------------------------------------------
def start_run(
    folder_name: str,
    material_code: str,
    *,
    product_prefix: str = "",
    dry_run: bool = False,
    settings: Settings | None = None,
    synchronous: bool = False,
) -> dict[str, Any]:
    """Create the batch_run row and process the folder.

    ``synchronous=True`` is used by the tests; the API runs the work on a worker thread
    so the progress bar has something to poll.
    """
    settings = settings or get_settings()
    folder = resolve_folder(folder_name, settings)
    files = image_files(folder)
    now = datetime.now(UTC).astimezone().isoformat(timespec="milliseconds")

    conn = connect(settings.db_file)
    try:
        material_id = materials.material_id(conn, material_code)
        batch_run_id = batches.create(
            conn,
            {
                "source_folder": str(folder),
                "material_id": material_id,
                "started_at": now,
                "status": "dry_run" if dry_run else "running",
                "image_count": len(files),
            },
        )
        conn.commit()
    finally:
        conn.close()

    _set_progress(
        batch_run_id,
        run_id=batch_run_id,
        total=len(files),
        processed=0,
        failures=0,
        current_file=None,
        status="dry_run" if dry_run else "running",
        started_at=time.time(),
        eta_seconds=None,
    )

    if dry_run:
        conn = connect(settings.db_file)
        try:
            batches.finish(
                conn,
                batch_run_id,
                {
                    "finished_at": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                    "image_count": len(files),
                    "status": "dry_run",
                },
            )
            conn.commit()
        finally:
            conn.close()
        _set_progress(batch_run_id, status="dry_run", processed=0)
        return {"batch_run_id": batch_run_id, "dry_run": True, "image_count": len(files)}

    args = (batch_run_id, files, material_code, product_prefix, settings)
    if synchronous:
        _process(*args)
    else:
        thread = threading.Thread(target=_process, args=args, daemon=True, name=f"batch-{batch_run_id}")
        thread.start()

    return {"batch_run_id": batch_run_id, "dry_run": False, "image_count": len(files)}


def _process(
    batch_run_id: int,
    files: list[Path],
    material_code: str,
    product_prefix: str,
    settings: Settings,
) -> None:
    provider = get_provider(settings)
    conn = connect(settings.db_file)
    started = time.time()
    failures = 0
    try:
        for index, path in enumerate(files, start=1):
            _set_progress(batch_run_id, current_file=path.name)
            try:
                image_bytes = path.read_bytes()
            except OSError:
                image_bytes = b""

            product_id = f"{product_prefix}{path.stem}" if product_prefix else path.stem
            result = provider.inspect(
                None,
                image_bytes,
                product_id=product_id,
                material=material_code,
                key=f"batch:{path.name}",
                station_id=settings.station_id,
            )
            # Keep the batch file itself as the stored source image where the provider
            # did not write one (a failed decode has no decoded image to save).
            if not result.source_image_path and path.exists():
                result.source_image_path = str(path)

            inspection_service.store_result(
                conn, result, batch_run_id=batch_run_id, settings=settings
            )
            conn.commit()

            if result.is_failure:
                failures += 1
            elapsed = time.time() - started
            per_item = elapsed / index if index else 0.0
            _set_progress(
                batch_run_id,
                processed=index,
                failures=failures,
                eta_seconds=round(per_item * (len(files) - index), 1),
            )

        totals = compute_totals(conn, batch_run_id)
        batches.finish(
            conn,
            batch_run_id,
            {
                "finished_at": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                "image_count": totals["processed"],
                "clean_count": totals["clean"],
                "regions_found_count": totals["regions_found"],
                "failure_count": totals["failed"],
                "status": "completed",
            },
        )
        conn.commit()
        _set_progress(batch_run_id, status="completed", processed=len(files), eta_seconds=0)
    except Exception as exc:  # pragma: no cover - defensive
        _set_progress(batch_run_id, status="failed", error=str(exc))
        try:
            batches.finish(
                conn,
                batch_run_id,
                {
                    "finished_at": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                    "status": "failed",
                },
            )
            conn.commit()
        except sqlite3.Error:
            pass
    finally:
        conn.close()


# -- reporting ------------------------------------------------------------------
def batch_filters(batch_run_id: int, **overrides: Any) -> HistoryFilters:
    return HistoryFilters(batch_run_id=batch_run_id, page_size=200, **overrides)


def compute_totals(conn: sqlite3.Connection, batch_run_id: int, status: str | None = None) -> dict[str, Any]:
    """Summary cards, computed from the same rows the export produces."""
    filters = batch_filters(batch_run_id, status=status)
    statuses = inspections.status_totals(conn, filters)
    regions = inspections.region_totals(conn, filters)
    processed = statuses["processed"]
    failed = statuses["acquisition_failure"] + statuses["processing_failure"]
    return {
        "processed": processed,
        "regions_found": statuses["regions_found"],
        "clean": statuses["clean"],
        "acquisition_failure": statuses["acquisition_failure"],
        "processing_failure": statuses["processing_failure"],
        "failed": failed,
        "regions_total": regions["total"],
        "crack_regions": regions["crack"],
        "scratch_regions": regions["scratch"],
        "regions_pct": round(statuses["regions_found"] / processed * 100, 1) if processed else 0.0,
        "clean_pct": round(statuses["clean"] / processed * 100, 1) if processed else 0.0,
    }


def results(
    conn: sqlite3.Connection,
    batch_run_id: int,
    *,
    status: str | None = None,
    only_with_regions: bool = False,
) -> list[dict[str, Any]]:
    filters = batch_filters(batch_run_id, status="regions_found" if only_with_regions else status)
    rows = inspections.all_matching(conn, filters)
    for row in rows:
        row["class_breakdown"] = inspections.class_breakdown(conn, row["inspection_id"])
    return rows


def report(
    conn: sqlite3.Connection,
    batch_run_id: int,
    *,
    status: str | None = None,
    only_with_regions: bool = False,
) -> dict[str, Any] | None:
    run = batches.get(conn, batch_run_id)
    if run is None:
        return None
    effective_status = "regions_found" if only_with_regions else status
    return {
        "run": run,
        "totals": compute_totals(conn, batch_run_id),
        "filtered_totals": compute_totals(conn, batch_run_id, status=effective_status),
        "rows": results(conn, batch_run_id, status=status, only_with_regions=only_with_regions),
        "progress": get_progress(batch_run_id),
        "filter": {"status": status, "only_with_regions": only_with_regions},
    }
