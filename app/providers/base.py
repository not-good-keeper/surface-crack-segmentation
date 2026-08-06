"""The integration boundary between the application and the inspection pipeline.

Everything above this line (services, routes, templates) is written against
``InspectionProvider``.  Everything below it is either the deterministic mock or the
real ``app.inference.Inspector``.  Swapping one for the other is a configuration
change, not a redesign.

The frontend and the service layer must never call ONNX Runtime, never re-implement
preprocessing, and never apply their own thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Inspection status values.  Stored verbatim in inspection.status.
STATUS_REGIONS_FOUND = "regions_found"
STATUS_CLEAN = "clean"
STATUS_ACQUISITION_FAILURE = "acquisition_failure"
STATUS_PROCESSING_FAILURE = "processing_failure"

ALL_STATUSES = (
    STATUS_REGIONS_FOUND,
    STATUS_CLEAN,
    STATUS_ACQUISITION_FAILURE,
    STATUS_PROCESSING_FAILURE,
)

#: Model output channel -> defect class code.  Channel 0 is background.
CLASS_BY_CHANNEL = {1: "crack", 2: "scratch"}
CHANNEL_BY_CLASS = {v: k for k, v in CLASS_BY_CHANNEL.items()}


@dataclass
class RegionRecord:
    """One detected region, in application units.

    ``length_px`` is medial-axis arc length with diagonal steps counted as sqrt(2).
    ``max_width_px`` is twice the maximum distance-transform value along the skeleton,
    i.e. the widest *inscribable* point, not the widest visual extent.
    """

    region_index: int
    class_code: str
    area_px: int
    length_px: float
    max_width_px: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    centroid: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_index": self.region_index,
            "class_code": self.class_code,
            "area_px": self.area_px,
            "length_px": round(self.length_px, 2),
            "max_width_px": round(self.max_width_px, 2),
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "centroid": (
                {"x": round(self.centroid[0], 2), "y": round(self.centroid[1], 2)}
                if self.centroid
                else None
            ),
        }


@dataclass
class InspectionResult:
    """The canonical result handed back by any provider.

    ``record`` is the JSON record that is appended to the record log; the remaining
    fields are what the application needs to persist and render.  The overlay is
    always at SOURCE resolution, never 256x256.
    """

    status: str
    empty: bool
    regions: list[RegionRecord] = field(default_factory=list)
    product_id: str | None = None
    material: str | None = None
    station_id: str | None = None
    image_sha256: str | None = None
    captured_at: str | None = None
    processed_at: str | None = None
    latency_ms: float | None = None
    source_image_path: str | None = None
    overlay_image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def is_failure(self) -> bool:
        return self.status in (STATUS_ACQUISITION_FAILURE, STATUS_PROCESSING_FAILURE)


@runtime_checkable
class InspectionProvider(Protocol):
    """Contract implemented by MockInspectionProvider and RealInspectionProvider."""

    name: str

    def inspect(
        self,
        image_bgr: Any,
        image_bytes: bytes | None,
        product_id: str,
        material: str,
    ) -> InspectionResult:
        """Inspect one image and return a canonical result."""
        ...

    def describe(self) -> dict[str, Any]:
        """Provider metadata for the status screen (never used for decisions)."""
        ...


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot run - missing model file, hash mismatch, etc.

    The application surfaces this on the Status screen and blocks inspection rather
    than producing results from a file it cannot identify.
    """

    def __init__(self, message: str, *, code: str = "provider_unavailable") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
