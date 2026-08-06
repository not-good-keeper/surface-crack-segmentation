"""Typed API models for material coverage, thresholds and model metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MaterialOut(BaseModel):
    material_code: str
    material_name: str
    support_status: str
    support_label: str
    is_limited: bool
    training_masks: int | None = None
    training_masks_display: str | None = None
    class_typing: float | None = None
    notes: str | None = None


class ThresholdsOut(BaseModel):
    stored: dict[str, Any]
    module: dict[str, Any]
    in_sync: bool
    source: str


class MaterialsOut(BaseModel):
    materials: list[MaterialOut]
    thresholds: ThresholdsOut
    model: dict[str, Any]
    overall: dict[str, Any]
