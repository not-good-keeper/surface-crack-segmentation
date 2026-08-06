"""Typed API models for inspections and regions.

Note what is absent: there is no score, probability or confidence field anywhere in
these schemas.  The model's scores are uncalibrated, so the API does not offer the
frontend a number it could render as "87 % confident" (NFR-09 / T-21).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

InspectionStatus = Literal["regions_found", "clean", "acquisition_failure", "processing_failure"]
BannerState = Literal["regions_found", "clean", "could_not_process", "check_station"]


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Centroid(BaseModel):
    x: float | None = None
    y: float | None = None


class RegionOut(BaseModel):
    region_index: int
    class_code: str = Field(description="crack or scratch - provisional, see material coverage")
    display_name: str | None = None
    area_px: int
    length_px: float = Field(description="medial-axis arc length, diagonal steps counted as sqrt(2)")
    max_width_px: float = Field(description="widest inscribable point, not the widest visual extent")
    bbox: BoundingBox
    centroid: Centroid | None = None


class ModelOut(BaseModel):
    version: str | None = None
    file_name: str | None = None
    artefact_sha256: str | None = None


class ProfileOut(BaseModel):
    version_no: int | None = None
    crack_threshold: float | None = None
    scratch_threshold: float | None = None
    minimum_area_px: int | None = None
    minimum_skeleton_px: int | None = None


class SummaryOut(BaseModel):
    state: str
    headline: str
    detail: str | None = None
    reason: str | None = None


class InspectionOut(BaseModel):
    inspection_id: int
    product_id: str | None = None
    material: str | None = None
    material_support_status: str | None = None
    station: str | None = None
    captured_at: str
    processed_at: str | None = None
    status: InspectionStatus
    region_count: int
    latency_ms: float | None = None
    image_sha256: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    batch_run_id: int | None = None
    max_length_px: float | None = None
    model: ModelOut
    profile: ProfileOut
    summary: SummaryOut
    regions: list[RegionOut] = []
    source_image_url: str | None = None
    overlay_image_url: str | None = None


class InspectionListOut(BaseModel):
    total: int
    page: int
    page_size: int
    page_count: int
    filters: dict[str, Any]
    totals: dict[str, int]
    items: list[dict[str, Any]]


class ErrorOut(BaseModel):
    error: str
    code: str
    detail: str | None = None
