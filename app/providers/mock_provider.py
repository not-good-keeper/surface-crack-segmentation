"""Deterministic mock inspection provider.

Produces realistic inspections - clean parts, crack regions, scratch regions, mixed
regions, acquisition failures and processing failures - with no ONNX file present.

Determinism: every result is a pure function of ``(mock_seed, key)``.  The same key
always yields the same status, the same regions, the same geometry and the same pixels.
``tests/test_mock_provider.py`` asserts this.

What the mock is NOT: it is not a model and its geometry is not a measurement of one.
Numbers produced here must never be quoted as accuracy.  The mock exists so the
interface, the database, the exports and the tests can be finished and reviewed before
the final weights land, and so that the day the weights arrive only the provider
changes.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.profiles import ACTIVE_PROFILE
from app.providers import mock_assets as assets
from app.providers.base import (
    STATUS_ACQUISITION_FAILURE,
    STATUS_CLEAN,
    STATUS_PROCESSING_FAILURE,
    STATUS_REGIONS_FOUND,
    InspectionResult,
    RegionRecord,
)

MATERIAL_ROTATION = ["steel", "steel", "steel", "plastic", "ceramic", "epoxy", "glass", "non_steel_metal"]

ACQUISITION_ERRORS = [
    ("image_decode_failed", "The file could not be decoded as an image."),
    ("truncated_file", "The image file ended before the last row of pixels."),
    ("zero_byte_file", "The file is zero bytes."),
    ("unsupported_media_type", "The extension says PNG but the magic bytes do not."),
    ("camera_timeout", "The camera did not deliver a frame within the capture window."),
]

PROCESSING_ERRORS = [
    ("inference_failed", "The inference session raised an error while running the graph."),
    ("postprocess_failed", "Region measurement failed after the class map was produced."),
    ("overlay_write_failed", "The overlay could not be written to the overlay folder."),
]

MAX_DECODED_PIXELS = 40_000_000  # decode size limit, rejects a decompression bomb


def _safe_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:80]


class MockInspectionProvider:
    """Implements ``InspectionProvider`` without any model file."""

    name = "mock"

    def __init__(self, settings: Settings | None = None, seed: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.seed = self.settings.mock_seed if seed is None else seed
        self.station_id = self.settings.station_id
        self.source_dir = self.settings.source_dir / "mock"
        self.overlay_dir = self.settings.overlay_dir / "mock"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

    # -- scenario ---------------------------------------------------------------
    def scenario(self, key: str, force_kind: str | None = None) -> dict[str, Any]:
        """Decide what this key's inspection looks like.  Pure function of the key."""
        rng = assets.rng_for(f"scenario:{key}", self.seed)
        roll = float(rng.random())

        if force_kind:
            kind = force_kind
        elif roll < 0.055:
            kind = "acquisition_failure"
        elif roll < 0.085:
            kind = "processing_failure"
        elif roll < 0.42:
            kind = "clean"
        elif roll < 0.62:
            kind = "crack_only"
        elif roll < 0.80:
            kind = "scratch_only"
        else:
            kind = "mixed"

        crack_count = scratch_count = 0
        if kind == "crack_only":
            crack_count = int(rng.integers(1, 4))
        elif kind == "scratch_only":
            scratch_count = int(rng.integers(1, 4))
        elif kind == "mixed":
            crack_count = int(rng.integers(1, 3))
            scratch_count = int(rng.integers(1, 3))

        size = assets.SOURCE_SIZES[int(rng.integers(0, len(assets.SOURCE_SIZES)))]
        cap = getattr(self.settings, "mock_image_max_width", 0) or 0
        if cap and size[0] > cap:
            # Scale the whole frame, keeping the aspect ratio. Used only to keep the
            # deployed demo bundle small; a real station never resizes its captures.
            size = (cap, max(1, round(size[1] * cap / size[0])))
        latency = float(round(rng.uniform(23.5, 31.5), 1))

        error_code = error_message = None
        if kind == "acquisition_failure":
            error_code, error_message = ACQUISITION_ERRORS[int(rng.integers(0, len(ACQUISITION_ERRORS)))]
        elif kind == "processing_failure":
            error_code, error_message = PROCESSING_ERRORS[int(rng.integers(0, len(PROCESSING_ERRORS)))]

        return {
            "kind": kind,
            "crack_count": crack_count,
            "scratch_count": scratch_count,
            "size": size,
            "latency_ms": latency,
            "error_code": error_code,
            "error_message": error_message,
        }

    def material_for(self, key: str) -> str:
        rng = assets.rng_for(f"material:{key}", self.seed)
        return MATERIAL_ROTATION[int(rng.integers(0, len(MATERIAL_ROTATION)))]

    # -- provider contract ------------------------------------------------------
    def inspect(
        self,
        image_bgr: Any,
        image_bytes: bytes | None,
        product_id: str,
        material: str,
        *,
        key: str | None = None,
        force_kind: str | None = None,
        captured_at: datetime | None = None,
        station_id: str | None = None,
    ) -> InspectionResult:
        """Inspect one image.

        ``image_bgr`` / ``image_bytes`` may both be ``None`` in mock mode, in which case
        a surface is synthesised.  When bytes ARE supplied they are decoded for real, so
        a genuinely broken file produces a genuine acquisition failure rather than a
        simulated one.
        """
        key = key or product_id
        scen = self.scenario(key, force_kind=force_kind)
        captured = captured_at or datetime.now(UTC).astimezone()
        station = station_id or self.station_id
        processed = captured + timedelta(milliseconds=scen["latency_ms"])

        base = {
            "product_id": product_id,
            "material": material,
            "station_id": station,
            "captured_at": captured.isoformat(timespec="milliseconds"),
            "processed_at": processed.isoformat(timespec="milliseconds"),
            "latency_ms": scen["latency_ms"],
        }

        # 1. Acquisition: decode the supplied bytes, or synthesise a surface.
        source_image: Image.Image | None = None
        if image_bytes is not None:
            decoded, err = self._decode(image_bytes)
            if err is not None:
                return self._failure(
                    STATUS_ACQUISITION_FAILURE, err[0], err[1], key=key,
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(), **base
                )
            source_image = decoded
        elif image_bgr is not None:
            arr = np.asarray(image_bgr)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return self._failure(
                    STATUS_ACQUISITION_FAILURE, "image_decode_failed",
                    "The supplied array is not an HxWx3 BGR image.", key=key, **base
                )
            source_image = Image.fromarray(arr[:, :, ::-1].astype(np.uint8), mode="RGB")
        elif scen["kind"] == "acquisition_failure":
            return self._failure(
                STATUS_ACQUISITION_FAILURE, scen["error_code"], scen["error_message"],
                key=key, **base
            )

        if source_image is None:
            rng = assets.rng_for(f"surface:{key}", self.seed)
            source_image = assets.make_surface(rng, material, scen["size"])
            regions = self._regions_for(key, source_image.size, scen)
            source_image = assets.burn_defects(source_image, regions)
        else:
            regions = self._regions_for(key, source_image.size, scen)

        # 2. Processing failure happens after acquisition succeeded - a different state.
        if scen["kind"] == "processing_failure":
            source_path = self._save(source_image, self.source_dir, f"{_safe_key(key)}_src.png")
            return self._failure(
                STATUS_PROCESSING_FAILURE, scen["error_code"], scen["error_message"],
                key=key, source_image_path=source_path,
                image_sha256=self._sha_of(source_image), **base
            )

        # 3. Regions and overlay, at source resolution.
        overlay_image = assets.draw_overlay(source_image, regions)
        source_path = self._save(source_image, self.source_dir, f"{_safe_key(key)}_src.png")
        overlay_path = self._save(overlay_image, self.overlay_dir, f"{_safe_key(key)}_ovl.png")

        region_records = [
            RegionRecord(
                region_index=i,
                class_code=r.class_code,
                area_px=r.area_px,
                length_px=r.length_px,
                max_width_px=r.max_width_px,
                bbox=r.bbox,
                centroid=r.centroid,
            )
            for i, r in enumerate(regions, start=1)
        ]

        status = STATUS_REGIONS_FOUND if region_records else STATUS_CLEAN
        result = InspectionResult(
            status=status,
            empty=not region_records,
            regions=region_records,
            image_sha256=self._sha_of(source_image),
            source_image_path=source_path,
            overlay_image_path=overlay_path,
            image_width=source_image.width,
            image_height=source_image.height,
            **base,
        )
        result.record = self._record(result)
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "deterministic": True,
            "seed": self.seed,
            "model_required": False,
            "note": "Mock results are generated, not measured. They demonstrate the interface only.",
        }

    # -- helpers ----------------------------------------------------------------
    def _regions_for(self, key: str, size: tuple[int, int], scen: dict[str, Any]) -> list[assets.MockRegion]:
        if scen["crack_count"] == 0 and scen["scratch_count"] == 0:
            return []
        rng = assets.rng_for(f"paths:{key}", self.seed)
        paths = assets.make_defect_paths(rng, size, scen["crack_count"], scen["scratch_count"])
        return assets.measure_paths(paths, size)

    def _decode(self, image_bytes: bytes) -> tuple[Image.Image | None, tuple[str, str] | None]:
        """Real decode with real failure modes (extension/magic/size are checked)."""
        if not image_bytes:
            return None, ("zero_byte_file", "The file is zero bytes.")
        try:
            probe = Image.open(io.BytesIO(image_bytes))
            width, height = probe.size
            if width * height > MAX_DECODED_PIXELS:
                return None, (
                    "oversized_image",
                    f"Decoded size {width}x{height} exceeds the configured decode limit.",
                )
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # forces the truncated-file error to surface here
            return img.convert("RGB"), None
        except UnidentifiedImageError:
            return None, ("unsupported_media_type", "The file is not a recognised image format.")
        except OSError as exc:
            return None, ("truncated_file", f"The image could not be fully read: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            return None, ("image_decode_failed", f"Decode failed: {exc}")

    def _failure(
        self,
        status: str,
        error_code: str | None,
        error_message: str | None,
        *,
        key: str,
        image_sha256: str | None = None,
        source_image_path: str | None = None,
        **base: Any,
    ) -> InspectionResult:
        result = InspectionResult(
            status=status,
            empty=False,  # NOT empty: an unreadable image is not a clean part
            regions=[],
            error_code=error_code,
            error_message=error_message,
            image_sha256=image_sha256,
            source_image_path=source_image_path,
            **base,
        )
        result.record = self._record(result)
        return result

    @staticmethod
    def _sha_of(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @staticmethod
    def _save(image: Image.Image, folder: Path, name: str) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        image.save(path, format="PNG", optimize=True)
        return str(path)

    def _record(self, result: InspectionResult) -> dict[str, Any]:
        """The canonical inspection record (the JSON appended to the record log).

        The real provider maps ``Inspector``'s record onto these same keys, so nothing
        downstream - database writer, exports, templates - has to know which provider
        produced a record.
        """
        return {
            "schema_version": 1,
            "provider": self.name,
            "station_id": result.station_id,
            "product_id": result.product_id,
            "material": result.material,
            "captured_at": result.captured_at,
            "processed_at": result.processed_at,
            "image_sha256": result.image_sha256,
            "image": {"width": result.image_width, "height": result.image_height},
            "status": result.status,
            "empty": result.empty,
            "latency_ms": result.latency_ms,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "profile": ACTIVE_PROFILE.as_dict(),
            "regions": [r.as_dict() for r in result.regions],
        }
