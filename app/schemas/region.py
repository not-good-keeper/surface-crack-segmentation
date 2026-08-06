"""Region detail payload."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.inspection import ModelOut, ProfileOut, RegionOut


class RegionDetailOut(BaseModel):
    inspection_id: int
    region: RegionOut
    region_count: int
    prev_index: int | None = None
    next_index: int | None = None
    product_id: str | None = None
    station: str | None = None
    material: str | None = None
    captured_at: str | None = None
    image_sha256: str | None = None
    model: ModelOut
    profile: ProfileOut
    crop_url: str | None = None
    measurement_note: str = (
        "Maximum width is the widest inscribable point measured along the region "
        "skeleton. It may differ from the widest visual extent."
    )
