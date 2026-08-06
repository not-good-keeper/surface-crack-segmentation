"""CSV and JSON exports.

Exports are generated from the same repository query that feeds the on-screen table
and the summary cards, so an exported total always matches a displayed total (T-16).

One row per region, plus one row per clean image, plus one row per failure carrying its
status - a failure is never emitted as a clean row.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.repositories import inspection_repository as inspections
from app.repositories.inspection_repository import HistoryFilters

CSV_COLUMNS = [
    "inspection_id",
    "captured_at",
    "product_id",
    "material",
    "station",
    "status",
    "region_count",
    "region_index",
    "region_class",
    "area_px",
    "length_px",
    "max_width_px",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "centroid_x",
    "centroid_y",
    "max_length_px",
    "latency_ms",
    "model_version",
    "model_sha256",
    "profile_version",
    "crack_threshold",
    "scratch_threshold",
    "minimum_area_px",
    "minimum_skeleton_px",
    "image_sha256",
    "error_code",
    "error_message",
    "batch_run_id",
]


def _rows_with_regions(conn: sqlite3.Connection, filters: HistoryFilters) -> list[dict[str, Any]]:
    out = []
    for row in inspections.all_matching(conn, filters):
        regions = inspections.regions(conn, row["inspection_id"])
        out.append({"inspection": row, "regions": regions})
    return out


def to_csv(conn: sqlite3.Connection, filters: HistoryFilters) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for item in _rows_with_regions(conn, filters):
        row = item["inspection"]
        base = {
            "inspection_id": row["inspection_id"],
            "captured_at": row["captured_at"],
            "product_id": row["product_id"],
            "material": row["material_code"],
            "station": row["station_code"],
            "status": row["status"],
            "region_count": row["region_count"],
            "max_length_px": row["max_length_px"],
            "latency_ms": row["latency_ms"],
            "model_version": row["model_version"],
            "model_sha256": row["artefact_sha256"],
            "profile_version": row["profile_version"],
            "crack_threshold": row["crack_threshold"],
            "scratch_threshold": row["scratch_threshold"],
            "minimum_area_px": row["minimum_area_px"],
            "minimum_skeleton_px": row["minimum_skeleton_px"],
            "image_sha256": row["image_sha256"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "batch_run_id": row["batch_run_id"],
        }
        if item["regions"]:
            for region in item["regions"]:
                writer.writerow(
                    {
                        **base,
                        "region_index": region["region_index"],
                        "region_class": region["class_code"],
                        "area_px": region["area_px"],
                        "length_px": region["length_px"],
                        "max_width_px": region["max_width_px"],
                        "bbox_x": region["bbox_x"],
                        "bbox_y": region["bbox_y"],
                        "bbox_width": region["bbox_width"],
                        "bbox_height": region["bbox_height"],
                        "centroid_x": region["centroid_x"],
                        "centroid_y": region["centroid_y"],
                    }
                )
        else:
            writer.writerow(base)
    return buffer.getvalue()


def to_json(conn: sqlite3.Connection, filters: HistoryFilters, *, context: dict[str, Any] | None = None) -> str:
    items = _rows_with_regions(conn, filters)
    payload = {
        "exported_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "filters": filters.active(),
        "totals": inspections.status_totals(conn, filters),
        "region_totals": inspections.region_totals(conn, filters),
        "count": len(items),
        "context": context or {},
        "inspections": [
            {
                "inspection_id": i["inspection"]["inspection_id"],
                "captured_at": i["inspection"]["captured_at"],
                "processed_at": i["inspection"]["processed_at"],
                "product_id": i["inspection"]["product_id"],
                "material": i["inspection"]["material_code"],
                "station": i["inspection"]["station_code"],
                "status": i["inspection"]["status"],
                "region_count": i["inspection"]["region_count"],
                "latency_ms": i["inspection"]["latency_ms"],
                "image_sha256": i["inspection"]["image_sha256"],
                "error_code": i["inspection"]["error_code"],
                "error_message": i["inspection"]["error_message"],
                "batch_run_id": i["inspection"]["batch_run_id"],
                "model": {
                    "version": i["inspection"]["model_version"],
                    "file_name": i["inspection"]["model_file_name"],
                    "artefact_sha256": i["inspection"]["artefact_sha256"],
                },
                "profile": {
                    "version_no": i["inspection"]["profile_version"],
                    "crack_threshold": i["inspection"]["crack_threshold"],
                    "scratch_threshold": i["inspection"]["scratch_threshold"],
                    "minimum_area_px": i["inspection"]["minimum_area_px"],
                    "minimum_skeleton_px": i["inspection"]["minimum_skeleton_px"],
                },
                "regions": [
                    {
                        "region_index": r["region_index"],
                        "class_code": r["class_code"],
                        "area_px": r["area_px"],
                        "length_px": r["length_px"],
                        "max_width_px": r["max_width_px"],
                        "bbox": {
                            "x": r["bbox_x"],
                            "y": r["bbox_y"],
                            "width": r["bbox_width"],
                            "height": r["bbox_height"],
                        },
                        "centroid": {"x": r["centroid_x"], "y": r["centroid_y"]},
                    }
                    for r in i["regions"]
                ],
            }
            for i in items
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
