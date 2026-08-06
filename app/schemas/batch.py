"""Typed API models for batch runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatchStartIn(BaseModel):
    source_folder: str = Field(description="A folder inside BATCH_ROOT. Absolute paths outside it are refused.")
    material: str
    product_prefix: str = ""
    dry_run: bool = False


class BatchTotals(BaseModel):
    processed: int
    regions_found: int
    clean: int
    acquisition_failure: int
    processing_failure: int
    failed: int
    regions_total: int
    crack_regions: int
    scratch_regions: int
    regions_pct: float
    clean_pct: float


class BatchProgress(BaseModel):
    total: int = 0
    processed: int = 0
    failures: int = 0
    current_file: str | None = None
    status: str = "running"
    eta_seconds: float | None = None


class BatchOut(BaseModel):
    batch_run_id: int
    source_folder: str
    material: str | None = None
    started_at: str
    finished_at: str | None = None
    status: str
    totals: BatchTotals
    progress: BatchProgress | None = None
    rows: list[dict[str, Any]] = []
