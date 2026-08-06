"""Adapter onto the real inference core, ``app/inference.py::Inspector``.

This module is the ONLY place in the application that knows the pipeline exists.  It
is imported lazily: nothing here runs, and ``onnxruntime`` is never imported, while
``INSPECTION_PROVIDER=mock``.

What this adapter deliberately does NOT do
------------------------------------------
- It does not import or call ``onnxruntime``.
- It does not resize, scale or normalise anything.
- It does not apply softmax.
- It does not compare a probability against a threshold.
- It does not label connected components or measure geometry.
- It does not down-scale the overlay: whatever resolution ``Inspector`` returns is
  written to disk unchanged.

All of the above already exist inside ``Inspector`` and are the exact path the reported
accuracy figures came from.  A second implementation that resized differently or used
argmax would be a different system with different accuracy that no benchmark would
catch (Phase 2 report 3.1, 8.3).  This adapter translates key names and writes files.
It performs no image mathematics of any kind.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.profiles import ACTIVE_PROFILE
from app.providers.base import (
    STATUS_ACQUISITION_FAILURE,
    STATUS_CLEAN,
    STATUS_PROCESSING_FAILURE,
    STATUS_REGIONS_FOUND,
    InspectionResult,
    ProviderUnavailable,
    RegionRecord,
)

#: Tolerant key lookup.  The application's canonical name -> names the record may use.
#:
#: The names the pipeline actually emits are listed FIRST, and they are the ones
#: `app/postprocess.py::extract_regions` writes today: `type`, `bbox_xywh` and `id`.
#: Getting these wrong is not a loud failure — `_pick` would fall through to its
#: default and every region would be reported as a crack at bounding box (0,0,0,0),
#: which looks like a plausible result rather than a bug. `tests/test_model_outputs.py`
#: runs a real record through this mapping for that reason.
REGION_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "class_code": ("type", "class_code", "class", "class_name", "label"),
    "area_px": ("area_px", "area"),
    "length_px": ("length_px", "length"),
    "max_width_px": ("max_width_px", "max_width", "width_px"),
    "bbox": ("bbox_xywh", "bbox", "bounding_box", "box"),
    "centroid": ("centroid", "center", "centre"),
    "region_index": ("id", "region_index", "index"),
}

CHUNK = 1024 * 1024


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick(mapping: dict[str, Any], canonical: str, default: Any = None) -> Any:
    for name in REGION_KEY_ALIASES.get(canonical, (canonical,)):
        if name in mapping:
            return mapping[name]
    return default


def decode_bgr(image_bytes: bytes) -> Any:
    """Bytes -> BGR array, or None if the file is not a readable image.

    cv2 rather than Pillow, because this is the decode the pipeline itself uses
    (`app/batch.py`) and a second decoder can differ in colour handling, EXIF rotation
    and subsampling. Imported here rather than at module scope: nothing in this file
    may load while the mock provider is the configured one.
    """
    import cv2
    import numpy as np

    array = np.frombuffer(image_bytes, np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


class RealInspectionProvider:
    """Wraps ``Inspector`` and maps its output onto the application schema."""

    name = "real"

    def __init__(self, settings: Settings | None = None, inspector: Any = None) -> None:
        self.settings = settings or get_settings()
        self.station_id = self.settings.station_id
        self.model_path = self.settings.model_file
        self.overlay_dir = self.settings.overlay_dir / "real"
        self.source_dir = self.settings.source_dir / "real"
        self._inspector = inspector
        self._model_sha256: str | None = None
        if inspector is None:
            self._load()
        else:
            self._model_sha256 = "injected"

    # -- start-up ---------------------------------------------------------------
    def _load(self) -> None:
        """Verify the model file, then import and construct ``Inspector``.

        Every failure here is fatal and explicit.  A hash mismatch stops the run rather
        than producing results from a file the system cannot identify (NFR-13 / T-23).
        """
        if not self.model_path.exists():
            raise ProviderUnavailable(
                f"INSPECTION_PROVIDER=real but no model file at {self.model_path}. "
                "Set MODEL_PATH, or run with INSPECTION_PROVIDER=mock.",
                code="model_file_missing",
            )

        actual = sha256_of_file(self.model_path)
        expected = (self.settings.model_sha256 or "").strip().lower()
        if expected and actual.lower() != expected:
            raise ProviderUnavailable(
                "Model hash mismatch. Inspection is stopped because the file on disk is "
                f"not the model this deployment was configured with.\n"
                f"  expected MODEL_SHA256={expected}\n  found    {actual}",
                code="model_hash_mismatch",
            )
        self._model_sha256 = actual

        try:
            from app.inference import Inspector  # noqa: PLC0415 - lazy on purpose
        except ImportError as exc:
            raise ProviderUnavailable(
                "INSPECTION_PROVIDER=real but app/inference.py (the Inspector class) is "
                f"not importable: {exc}. The inference core must be present in real mode.",
                code="inference_core_missing",
            ) from exc

        try:
            self._inspector = Inspector(str(self.model_path), station_id=self.station_id)
        except Exception as exc:
            raise ProviderUnavailable(
                f"Inspector failed to initialise from {self.model_path}: {exc}",
                code="inspector_init_failed",
            ) from exc

        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    # -- provider contract ------------------------------------------------------
    def inspect(
        self,
        image_bgr: Any,
        image_bytes: bytes | None,
        product_id: str,
        material: str,
        *,
        key: str | None = None,
        captured_at: datetime | None = None,
        station_id: str | None = None,
        **_ignored: Any,
    ) -> InspectionResult:
        captured = captured_at or datetime.now(UTC).astimezone()
        started = datetime.now(UTC)

        if image_bgr is None and image_bytes:
            # A capture or an upload arrives as bytes. Decode it exactly the way
            # app/batch.py does, so a frame submitted through the browser and the same
            # file processed by the batch CLI enter the pipeline identically.
            image_bgr = decode_bgr(image_bytes)
            if image_bgr is None:
                return InspectionResult(
                    status=STATUS_ACQUISITION_FAILURE,
                    # NOT empty: an unreadable capture is not a defect-free part.
                    empty=False,
                    product_id=product_id,
                    material=material,
                    station_id=station_id or self.station_id,
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                    captured_at=captured.isoformat(timespec="milliseconds"),
                    processed_at=datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                    latency_ms=round((datetime.now(UTC) - started).total_seconds() * 1000.0, 2),
                    error_code="image_decode_failed",
                    error_message="The submitted file could not be decoded as an image.",
                    record={"provider": self.name, "error_code": "image_decode_failed"},
                )

        try:
            record, overlay_bgr, class_map = self._inspector.inspect(
                image_bgr, image_bytes, product_id=product_id, material=material
            )
        except Exception as exc:
            elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
            return InspectionResult(
                status=STATUS_PROCESSING_FAILURE,
                empty=False,
                product_id=product_id,
                material=material,
                station_id=station_id or self.station_id,
                captured_at=captured.isoformat(timespec="milliseconds"),
                processed_at=datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                latency_ms=round(elapsed, 2),
                error_code="inference_failed",
                error_message=str(exc),
                record={"provider": self.name, "error": str(exc)},
            )

        return self._to_result(
            record,
            overlay_bgr,
            class_map,
            product_id=product_id,
            material=material,
            station_id=station_id or self.station_id,
            captured=captured,
            key=key or product_id,
            image_bytes=image_bytes,
            # The record carries no timing, so the adapter measures it. This is the
            # wall clock around Inspector.inspect(): preprocess, session run,
            # class map, geometry and overlay. It is not the 26 ms inference figure
            # from the benchmark and must not be quoted as one.
            measured_ms=(datetime.now(UTC) - started).total_seconds() * 1000.0,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "deterministic": True,
            "model_required": True,
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
        }

    # -- record mapping ---------------------------------------------------------
    def _to_result(
        self,
        record: dict[str, Any],
        overlay_bgr: Any,
        class_map: Any,
        *,
        product_id: str,
        material: str,
        station_id: str,
        captured: datetime,
        key: str,
        image_bytes: bytes | None,
        measured_ms: float | None = None,
    ) -> InspectionResult:
        """Map the canonical record onto the application schema.

        The record produced by ``Inspector`` is preserved verbatim in ``result.record``
        and written to the record log unchanged; the fields below are only the columns
        the database needs for filtering and display.
        """
        status = str(record.get("status") or "").strip()
        raw_regions = record.get("regions") or []

        if not status:
            status = STATUS_REGIONS_FOUND if raw_regions else STATUS_CLEAN
        if status not in (
            STATUS_REGIONS_FOUND,
            STATUS_CLEAN,
            STATUS_ACQUISITION_FAILURE,
            STATUS_PROCESSING_FAILURE,
        ):
            # An unknown status is never quietly downgraded to "clean".
            status = STATUS_PROCESSING_FAILURE

        regions: list[RegionRecord] = []
        for index, raw in enumerate(raw_regions, start=1):
            bbox = _pick(raw, "bbox", [0, 0, 0, 0])
            if isinstance(bbox, dict):
                bbox_t = (
                    int(bbox.get("x", 0)),
                    int(bbox.get("y", 0)),
                    int(bbox.get("width", bbox.get("w", 0))),
                    int(bbox.get("height", bbox.get("h", 0))),
                )
            else:
                seq = list(bbox) + [0, 0, 0, 0]
                bbox_t = (int(seq[0]), int(seq[1]), int(seq[2]), int(seq[3]))

            centroid = _pick(raw, "centroid")
            if isinstance(centroid, dict):
                centroid_t = (float(centroid.get("x", 0)), float(centroid.get("y", 0)))
            elif centroid is not None:
                seq = list(centroid)
                centroid_t = (float(seq[0]), float(seq[1]))
            else:
                centroid_t = None

            class_code = _pick(raw, "class_code", "crack")
            if isinstance(class_code, int):  # a channel index, not a name
                class_code = {1: "crack", 2: "scratch"}.get(class_code, "crack")

            regions.append(
                RegionRecord(
                    region_index=int(_pick(raw, "region_index", index)),
                    class_code=str(class_code),
                    area_px=int(_pick(raw, "area_px", 0)),
                    length_px=float(_pick(raw, "length_px", 0.0)),
                    max_width_px=float(_pick(raw, "max_width_px", 0.0)),
                    bbox=bbox_t,
                    centroid=centroid_t,
                )
            )

        latency_ms = (
            float(record["latency_ms"])
            if record.get("latency_ms") is not None
            else (round(measured_ms, 2) if measured_ms is not None else None)
        )

        overlay_path = self._write_overlay(overlay_bgr, key)
        source_path = self._write_source(image_bytes, key)
        height = width = None
        if overlay_bgr is not None:
            try:
                height, width = int(overlay_bgr.shape[0]), int(overlay_bgr.shape[1])
            except Exception:  # pragma: no cover - defensive
                pass

        return InspectionResult(
            status=status,
            empty=bool(record.get("empty", not regions)),
            regions=regions,
            product_id=str(record.get("product_id") or product_id),
            material=str(record.get("material") or material),
            station_id=str(record.get("station_id") or station_id),
            image_sha256=record.get("image_sha256"),
            captured_at=str(record.get("captured_at") or captured.isoformat(timespec="milliseconds")),
            processed_at=str(
                record.get("processed_at")
                or (captured + timedelta(milliseconds=latency_ms or 0)).isoformat(
                    timespec="milliseconds"
                )
            ),
            latency_ms=latency_ms,
            source_image_path=source_path,
            overlay_image_path=overlay_path,
            image_width=width,
            image_height=height,
            error_code=record.get("error_code"),
            error_message=record.get("error_message"),
            record={
                "schema_version": 1,
                "provider": self.name,
                "profile": ACTIVE_PROFILE.as_dict(),
                "model_sha256": self._model_sha256,
                **record,
            },
        )

    def _write_overlay(self, overlay_bgr: Any, key: str) -> str | None:
        """Persist the overlay at whatever resolution Inspector produced.

        No resize, no re-encode of the mask, no re-drawing: BGR to RGB channel order
        only, because PNG writers expect RGB.
        """
        if overlay_bgr is None:
            return None
        from PIL import Image  # local import keeps mock mode free of the dependency path

        arr = overlay_bgr[:, :, ::-1] if getattr(overlay_bgr, "ndim", 0) == 3 else overlay_bgr
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        path = self.overlay_dir / f"{_safe(key)}_ovl.png"
        Image.fromarray(arr).save(path, format="PNG")
        return str(path)

    def _write_source(self, image_bytes: bytes | None, key: str) -> str | None:
        if not image_bytes:
            return None
        self.source_dir.mkdir(parents=True, exist_ok=True)
        path = self.source_dir / f"{_safe(key)}_src.png"
        try:
            from PIL import Image

            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(path, format="PNG")
        except Exception:
            return None
        return str(path)


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:80]
