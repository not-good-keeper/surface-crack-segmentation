"""The record mapping between the pipeline and the application schema.

This is the seam the interface was written against before the pipeline was available,
and three of its key aliases were wrong: `extract_regions` emits `type`, `bbox_xywh` and
`id`, while the adapter looked for `class_code`, `bbox` and `region_index`.

Nothing raises when that happens. `_pick` falls through to its default, so every region
comes out as a **crack** at bounding box **(0, 0, 0, 0)** — a result that looks entirely
reasonable on screen, with sensible areas and lengths and a class label that is simply
wrong. No metric in the benchmark would show it, because the benchmark never goes
through this adapter.

Hence these tests. The first needs nothing installed and pins the key names. The second
builds a record with the real `extract_regions` and runs it through the real adapter, so
a change to either side has to break something.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.providers.base import STATUS_CLEAN, STATUS_REGIONS_FOUND
from app.providers.real_provider import RealInspectionProvider


class StubInspector:
    """Stands in for `Inspector`, returning a record the pipeline could have produced."""

    def __init__(self, record, overlay=None, class_map=None):
        self.record = record
        self.overlay = overlay
        self.class_map = class_map

    def inspect(self, image_bgr, image_bytes, product_id=None, material=None):
        return self.record, self.overlay, self.class_map


#: Stands in for an already-decoded frame. Any non-None value skips the adapter's
#: decode step, which is what these tests want: they exercise the record mapping that
#: happens *after* inference, not acquisition.
FRAME = object()


def provider(settings, record):
    return RealInspectionProvider(settings, inspector=StubInspector(record))


def pipeline_record(regions, **extra):
    """The shape `app/postprocess.py::build_record` actually returns."""
    base = {
        "schema_version": "1.0",
        "image_sha256": "a" * 64,
        "model_version": "smpslim_c3v5",
        "station_id": "line-1-cam-A",
        "product_id": "batch-77/item-12",
        "material": "steel",
        "processed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "processing_profile": "conveyor-v3",
        "regions": regions,
        "empty": not regions,
        "overlay_path": None,
    }
    base.update(extra)
    return base


# -- the key names ------------------------------------------------------------
def test_the_pipelines_own_key_names_survive_the_mapping(settings):
    record = pipeline_record([
        {"id": 1, "type": "scratch", "area_px": 2118, "length_px": 409.0,
         "max_width_px": 9.0, "bbox_xywh": [107, 14, 95, 204]},
        {"id": 2, "type": "crack", "area_px": 684, "length_px": 212.0,
         "max_width_px": 4.0, "bbox_xywh": [58, 11, 28, 97]},
    ])
    result = provider(settings, record).inspect(
        FRAME, b"bytes", product_id="batch-77/item-12", material="steel"
    )

    assert result.status == STATUS_REGIONS_FOUND
    assert [r.region_index for r in result.regions] == [1, 2]
    # The one that mattered: a scratch must not arrive as a crack.
    assert [r.class_code for r in result.regions] == ["scratch", "crack"]
    assert [r.bbox for r in result.regions] == [(107, 14, 95, 204), (58, 11, 28, 97)]
    assert [r.area_px for r in result.regions] == [2118, 684]
    assert [r.length_px for r in result.regions] == [409.0, 212.0]
    assert [r.max_width_px for r in result.regions] == [9.0, 4.0]


def test_a_bounding_box_is_never_silently_zero(settings):
    """The failure mode the alias bug produced, asserted directly."""
    record = pipeline_record([
        {"id": 1, "type": "crack", "area_px": 300, "length_px": 90.0,
         "max_width_px": 3.0, "bbox_xywh": [10, 20, 30, 40]},
    ])
    result = provider(settings, record).inspect(FRAME, b"x", product_id="p", material="steel")
    assert result.regions[0].bbox != (0, 0, 0, 0)


def test_a_clean_record_stays_clean_and_empty(settings):
    """`build_record` emits no status, so it is derived - and must not become a failure."""
    result = provider(settings, pipeline_record([])).inspect(
        FRAME, b"x", product_id="p", material="steel"
    )
    assert result.status == STATUS_CLEAN
    assert result.empty is True
    assert result.regions == []
    assert result.error_code is None


def test_latency_is_measured_when_the_record_carries_none(settings):
    """`build_record` has no latency field; a blank health strip is not acceptable."""
    result = provider(settings, pipeline_record([])).inspect(
        FRAME, b"x", product_id="p", material="steel"
    )
    assert result.latency_ms is not None
    assert result.latency_ms >= 0


def test_an_unknown_status_is_never_downgraded_to_clean(settings):
    result = provider(settings, pipeline_record([], status="something_new")).inspect(
        FRAME, b"x", product_id="p", material="steel"
    )
    assert result.status == "processing_failure"
    assert result.status != STATUS_CLEAN


def test_bytes_that_are_not_an_image_never_reach_the_model(settings):
    """A capture arrives as bytes, so the adapter decodes before inference.

    An undecodable file is an acquisition failure and stops there — `empty` stays False,
    because a file that could not be read is not a defect-free part.
    """
    pytest.importorskip("cv2", reason="inference core not installed")

    result = provider(settings, pipeline_record([])).inspect(
        None, b"this is not an image", product_id="p", material="steel"
    )
    assert result.status == "acquisition_failure"
    assert result.empty is False
    assert result.error_code == "image_decode_failed"
    assert result.regions == []


# -- against the real post-processing -----------------------------------------
def test_regions_from_the_real_extract_regions_map_correctly(settings):
    """End to end over the seam, using the pipeline's own geometry code.

    Skipped where the inference core's dependencies are absent, which is the mock
    deployment - the environment this seam does not exist in.
    """
    np = pytest.importorskip("numpy", reason="inference core not installed")
    postprocess = pytest.importorskip("app.postprocess", reason="inference core not installed")

    # A thick crack and a separate thick scratch, far enough apart to stay two regions.
    class_map = np.zeros((120, 200), np.int32)
    class_map[20:26, 20:120] = postprocess.CRACK
    class_map[70:78, 30:170] = postprocess.SCRATCH

    regions = postprocess.extract_regions(class_map, postprocess.ACTIVE_PROFILE)
    assert len(regions) == 2, f"fixture did not produce two regions: {regions}"

    record = pipeline_record(regions)
    result = provider(settings, record).inspect(FRAME, b"x", product_id="p", material="steel")

    assert {r.class_code for r in result.regions} == {"crack", "scratch"}
    for mapped, raw in zip(result.regions, regions, strict=True):
        assert mapped.class_code == raw["type"]
        assert mapped.area_px == raw["area_px"]
        assert mapped.bbox == tuple(raw["bbox_xywh"])
        assert mapped.region_index == raw["id"]
