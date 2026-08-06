"""Mock provider: determinism, states, geometry floors, real decode failures."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.profiles import ACTIVE_PROFILE
from app.providers.base import (
    STATUS_ACQUISITION_FAILURE,
    STATUS_CLEAN,
    STATUS_PROCESSING_FAILURE,
    STATUS_REGIONS_FOUND,
    InspectionProvider,
)
from app.providers.mock_provider import MockInspectionProvider

CAPTURED = datetime(2026, 8, 6, 11, 42, 7)


@pytest.fixture()
def provider(settings):
    return MockInspectionProvider(settings)


def test_mock_provider_satisfies_the_provider_protocol(provider):
    assert isinstance(provider, InspectionProvider)


def test_same_key_gives_an_identical_result(provider):
    first = provider.inspect(None, None, "p-1", "steel", key="determinism-1", captured_at=CAPTURED)
    second = provider.inspect(None, None, "p-1", "steel", key="determinism-1", captured_at=CAPTURED)

    assert first.status == second.status
    assert first.image_sha256 == second.image_sha256
    assert [r.as_dict() for r in first.regions] == [r.as_dict() for r in second.regions]
    assert first.latency_ms == second.latency_ms


def test_two_providers_with_the_same_seed_agree(settings):
    a = MockInspectionProvider(settings, seed=404)
    b = MockInspectionProvider(settings, seed=404)
    ra = a.inspect(None, None, "p", "steel", key="k", captured_at=CAPTURED)
    rb = b.inspect(None, None, "p", "steel", key="k", captured_at=CAPTURED)
    assert ra.image_sha256 == rb.image_sha256
    assert [r.as_dict() for r in ra.regions] == [r.as_dict() for r in rb.regions]


def test_a_different_seed_gives_a_different_result(settings):
    a = MockInspectionProvider(settings, seed=404).inspect(None, None, "p", "steel", key="k", captured_at=CAPTURED)
    b = MockInspectionProvider(settings, seed=99).inspect(None, None, "p", "steel", key="k", captured_at=CAPTURED)
    assert a.image_sha256 != b.image_sha256


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("clean", STATUS_CLEAN),
        ("crack_only", STATUS_REGIONS_FOUND),
        ("scratch_only", STATUS_REGIONS_FOUND),
        ("mixed", STATUS_REGIONS_FOUND),
        ("acquisition_failure", STATUS_ACQUISITION_FAILURE),
        ("processing_failure", STATUS_PROCESSING_FAILURE),
    ],
)
def test_every_state_can_be_produced(provider, kind, expected):
    result = provider.inspect(None, None, "p", "steel", key=f"state-{kind}", force_kind=kind)
    assert result.status == expected


def test_clean_and_failure_are_different_states(provider):
    """A defect-free part and an unreadable image must never collapse into one state."""
    clean = provider.inspect(None, None, "p", "steel", key="c", force_kind="clean")
    failed = provider.inspect(None, None, "p", "steel", key="f", force_kind="acquisition_failure")

    assert clean.status == STATUS_CLEAN
    assert clean.empty is True
    assert clean.regions == []
    assert clean.error_code is None

    assert failed.status == STATUS_ACQUISITION_FAILURE
    assert failed.empty is False  # not empty - it was never read
    assert failed.regions == []
    assert failed.error_code is not None


def test_crack_only_produces_no_scratch_regions(provider):
    result = provider.inspect(None, None, "p", "steel", key="crack", force_kind="crack_only")
    assert result.regions
    assert {r.class_code for r in result.regions} == {"crack"}


def test_mixed_produces_both_classes(provider):
    for attempt in range(12):
        result = provider.inspect(None, None, "p", "steel", key=f"mix-{attempt}", force_kind="mixed")
        if {r.class_code for r in result.regions} == {"crack", "scratch"}:
            return
    pytest.fail("no mixed result produced both classes")


def test_regions_respect_the_profile_floors(provider):
    """Region floors come from Profile, so nothing below them is ever reported."""
    for index in range(20):
        result = provider.inspect(None, None, "p", "steel", key=f"floor-{index}")
        for region in result.regions:
            assert region.area_px >= ACTIVE_PROFILE.min_area_px
            assert region.length_px >= ACTIVE_PROFILE.min_skeleton_px


def test_geometry_is_plausible_and_complete(provider):
    result = provider.inspect(None, None, "p", "steel", key="geometry", force_kind="mixed")
    for region in result.regions:
        x, y, w, h = region.bbox
        assert w > 0 and h > 0
        assert region.area_px <= w * h
        assert region.max_width_px > 0
        assert region.centroid is not None
        assert x <= region.centroid[0] <= x + w
        assert y <= region.centroid[1] <= y + h


def test_overlay_is_written_at_source_resolution(provider):
    from PIL import Image

    result = provider.inspect(None, None, "p", "steel", key="resolution", force_kind="mixed")
    overlay = Image.open(result.overlay_image_path)
    source = Image.open(result.source_image_path)
    assert overlay.size == source.size
    assert overlay.size != (256, 256)


def test_record_carries_provenance_and_no_score(provider):
    result = provider.inspect(None, None, "batch-77/item-12", "steel", key="record", force_kind="mixed")
    record = result.record
    assert record["product_id"] == "batch-77/item-12"
    assert record["station_id"]
    assert record["captured_at"]
    assert record["image_sha256"]
    assert record["profile"]["crack_thresh"] == ACTIVE_PROFILE.crack_thresh

    serialised = str(record).lower()
    for forbidden in ("confidence", "probability", "score"):
        assert forbidden not in serialised


def test_zero_byte_file_is_an_acquisition_failure(provider):
    result = provider.inspect(None, b"", "p", "steel", key="zero")
    assert result.status == STATUS_ACQUISITION_FAILURE
    assert result.error_code == "zero_byte_file"


def test_text_file_with_a_png_extension_is_an_acquisition_failure(provider):
    result = provider.inspect(None, b"not an image at all", "p", "steel", key="text")
    assert result.status == STATUS_ACQUISITION_FAILURE
    assert result.error_code == "unsupported_media_type"


def test_truncated_png_is_an_acquisition_failure(provider):
    truncated = bytes.fromhex("89504e470d0a1a0a0000000d49484452000001000000010008060000")
    result = provider.inspect(None, truncated, "p", "steel", key="truncated")
    assert result.status == STATUS_ACQUISITION_FAILURE
    assert result.error_code in ("truncated_file", "unsupported_media_type", "image_decode_failed")


def test_a_valid_image_is_decoded_and_inspected(provider, tmp_path):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (120, 120, 120)).save(buffer, format="PNG")
    result = provider.inspect(None, buffer.getvalue(), "p", "steel", key="valid", force_kind="mixed")

    assert result.status == STATUS_REGIONS_FOUND
    assert result.image_width == 320 and result.image_height == 240


def test_describe_states_that_results_are_generated(provider):
    described = provider.describe()
    assert described["provider"] == "mock"
    assert described["deterministic"] is True
    assert described["model_required"] is False
