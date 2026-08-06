"""Turning a provider result into stored rows, and running the live/demo inspection.

Nothing here decides what a defect is.  The provider produced the record; this module
resolves foreign keys, writes the inspection row and its regions, and appends the
canonical record to the record log.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.providers import get_provider
from app.providers.base import InspectionResult
from app.repositories import inspection_repository as inspections
from app.repositories import material_repository as materials
from app.repositories import model_repository as models

RECORD_LOG_NAME = "records.jsonl"


def _station_id(conn: sqlite3.Connection, station_code: str) -> int:
    row = models.station_by_code(conn, station_code)
    if row:
        return int(row["station_id"])
    now = datetime.now(UTC).isoformat(timespec="seconds")
    line = station_code.split("-cam")[0] if "-cam" in station_code else station_code
    cur = conn.execute(
        "INSERT INTO station (station_code, line_code, mm_per_pixel, camera_status, created_at) "
        "VALUES (?, ?, NULL, 'ok', ?)",
        [station_code, line, now],
    )
    return int(cur.lastrowid)


def append_record_log(settings: Settings, record: dict[str, Any]) -> None:
    """Append-only JSONL record log.

    The database is the queryable store; the log is the flat record the Phase 1 design
    called for and is never rewritten.
    """
    path = settings.export_dir.parent / RECORD_LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def store_result(
    conn: sqlite3.Connection,
    result: InspectionResult,
    *,
    batch_run_id: int | None = None,
    settings: Settings | None = None,
    write_log: bool = True,
) -> int:
    """Persist one provider result and return the new inspection_id."""
    settings = settings or get_settings()

    material_id = materials.material_id(conn, result.material)
    profile = materials.active_profile(conn, result.material)
    if profile is None:
        raise RuntimeError("No active profile in the database - run the seed first.")
    model = models.active_model(conn)
    if model is None:
        raise RuntimeError("No model_version row in the database - run the seed first.")

    inspection_id = inspections.insert(
        conn,
        {
            "station_id": _station_id(conn, result.station_id or settings.station_id),
            "profile_id": profile["profile_id"],
            "model_version_id": model["model_version_id"],
            "material_id": material_id,
            "batch_run_id": batch_run_id,
            "image_sha256": result.image_sha256,
            "product_id": result.product_id,
            "captured_at": result.captured_at,
            "processed_at": result.processed_at,
            "status": result.status,
            "region_count": result.region_count,
            "latency_ms": result.latency_ms,
            "source_image_path": portable_media_path(result.source_image_path, settings),
            "overlay_image_path": portable_media_path(result.overlay_image_path, settings),
            "error_code": result.error_code,
            "error_message": result.error_message,
        },
    )

    for region in result.regions:
        x, y, w, h = region.bbox
        inspections.insert_region(
            conn,
            inspection_id,
            {
                "region_index": region.region_index,
                "class_id": materials.class_id(conn, region.class_code),
                "area_px": region.area_px,
                "length_px": region.length_px,
                "max_width_px": region.max_width_px,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "centroid_x": region.centroid[0] if region.centroid else None,
                "centroid_y": region.centroid[1] if region.centroid else None,
            },
        )

    if write_log and result.record:
        record = dict(result.record)
        record["inspection_id"] = inspection_id
        append_record_log(settings, record)

    return inspection_id


def next_demo_inspection(
    conn: sqlite3.Connection,
    settings: Settings | None = None,
    *,
    force_kind: str | None = None,
) -> int:
    """Advance the live station by one part.

    The mock sequence is keyed on the number of inspections already stored, so a fresh
    database always replays the same sequence of parts in the same order.
    """
    settings = settings or get_settings()
    provider = get_provider(settings)

    index = inspections.total_count(conn)
    key = f"live-{index:05d}"
    material = getattr(provider, "material_for", lambda _k: "steel")(key)
    product_id = f"batch-{77 + index // 24}/item-{index % 24 + 1:02d}"

    result = provider.inspect(
        None,
        None,
        product_id=product_id,
        material=material,
        key=key,
        force_kind=force_kind,
        station_id=settings.station_id,
    )
    return store_result(conn, result, settings=settings)


#: Largest capture accepted from a browser.  A phone photograph is a few megabytes; a
#: 16 MB ceiling leaves generous headroom while keeping one request from exhausting the
#: worker.  Anything under this is passed to the provider even if it is obviously not an
#: image, because the provider is what turns an unreadable file into a recorded
#: acquisition failure — rejecting it here would lose the record instead of making one.
CAPTURE_MAX_BYTES = 16 * 1024 * 1024


class CaptureTooLarge(ValueError):
    """The submitted capture is larger than CAPTURE_MAX_BYTES."""

    code = "capture_too_large"


def capture_inspection(
    conn: sqlite3.Connection,
    settings: Settings | None = None,
    *,
    image_bytes: bytes,
    filename: str | None = None,
    material: str = "steel",
    product_id: str | None = None,
    source: str = "upload",
) -> int:
    """Inspect one frame submitted by the operator, store it, return the inspection id.

    Used by both capture paths — the device camera and the file picker — because they
    differ only in where the browser got the bytes. The frame goes through the same
    provider, the same record mapping and the same database writer as a station
    inspection, so a capture is not a second class of result: it appears in History, in
    the exports and in the batch totals exactly like anything else.

    The mock provider decodes the submitted bytes for real and then draws *generated*
    regions on them. That is a demonstration of the interface, never a measurement —
    the capture screen says so on every result while the provider is `mock`.
    """
    settings = settings or get_settings()
    if len(image_bytes) > CAPTURE_MAX_BYTES:
        # MiB on both sides of the sentence. Dividing the limit by 1e6 here and by
        # 1048576 in the template had the screen offering 16 MB and the error refusing
        # at 17.
        mib = 1024 * 1024
        raise CaptureTooLarge(
            f"The capture is {len(image_bytes) / mib:.1f} MB; the limit is "
            f"{CAPTURE_MAX_BYTES // mib} MB."
        )

    provider = get_provider(settings)
    digest = hashlib.sha256(image_bytes).hexdigest()

    # Keyed on the image, not on a counter: submitting the same photograph twice gives
    # the same mock scenario both times, which is what makes the demo reproducible.
    key = f"capture-{digest[:12]}"
    if not product_id:
        stem = Path(filename).stem if filename else ""
        stem = "".join(c for c in stem if c.isalnum() or c in "-_")[:32]
        product_id = f"{source}/{stem or digest[:8]}"

    result = provider.inspect(
        None,
        image_bytes,
        product_id=product_id,
        material=material,
        key=key,
        station_id=settings.station_id,
    )
    inspection_id = store_result(conn, result, settings=settings)
    return inspection_id


def inspection_payload(conn: sqlite3.Connection, inspection_id: int) -> dict[str, Any] | None:
    """Everything the Live, Region-detail and History-detail screens need."""
    row = inspections.get(conn, inspection_id)
    if row is None:
        return None
    region_rows = inspections.regions(conn, inspection_id)
    breakdown = inspections.class_breakdown(conn, inspection_id)
    return {
        "inspection": row,
        "regions": region_rows,
        "class_breakdown": breakdown,
        "summary": summarise(row, breakdown),
    }


def summarise(row: dict[str, Any], breakdown: dict[str, int]) -> dict[str, Any]:
    """The result banner.

    Four states, distinguished by wording, fill and border together - never by colour
    alone.  None of them is a verdict: the operator decides what to do.
    """
    status = row["status"]
    count = int(row.get("region_count") or 0)

    if status == "regions_found":
        parts = [f"{n} {code}" for code, n in sorted(breakdown.items()) if n]
        return {
            "state": "regions_found",
            "headline": f"{count} REGION{'S' if count != 1 else ''} FOUND",
            "detail": ", ".join(parts) if parts else "",
            "reason": None,
        }
    if status == "clean":
        return {
            "state": "clean",
            "headline": "NO DEFECTS FOUND",
            "detail": "Recorded with an empty region list",
            "reason": None,
        }
    return {
        "state": "could_not_process",
        "headline": "COULD NOT PROCESS",
        "detail": row.get("error_code") or status,
        "reason": row.get("error_message"),
    }


def portable_media_path(raw: str | None, settings: Settings | None = None) -> str | None:
    """Store an image path relative to its data root where possible.

    An absolute path pins a database to the machine that built it: the deployed demo is
    seeded during the build and read on a different filesystem, and a backup restored
    into a different folder would lose every image. Recording
    ``sources/mock/live-00001_src.png`` instead keeps the database portable.
    """
    if not raw:
        return None
    settings = settings or get_settings()
    path = Path(raw).resolve()
    for root in (settings.runtime_data_dir, settings.bundled_data_dir):
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def image_path_for(row: dict[str, Any], kind: str, settings: Settings | None = None) -> Path | None:
    """Resolve a stored image path, refusing anything outside the configured roots.

    The path comes from the database rather than the request, and it is still checked:
    a stored path is only trusted if it resolves inside one of the configured media
    roots (see tests/test_security.py). Relative paths are tried against the writable
    root first, then the read-only bundled one, so a runtime-generated image wins over a
    shipped one of the same name.
    """
    settings = settings or get_settings()
    raw = row.get("overlay_image_path" if kind == "overlay" else "source_image_path")
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [settings.runtime_data_dir / raw, settings.bundled_data_dir / raw]

    roots = settings.media_roots
    for option in candidates:
        resolved = option.resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            continue
        if resolved.exists():
            return resolved
    return None
