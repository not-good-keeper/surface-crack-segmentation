"""Procedural industrial-surface and defect-overlay generation for mock mode.

No image is ever downloaded.  Every surface is synthesised from a seeded numpy
generator, so the same key always produces the same pixels, on any machine, offline.

This module generates *demonstration assets*.  It is not an inference pipeline and it
does not re-implement one: there is no softmax here, no class-decision rule, no
connected-component labelling and no medial-axis transform.  Geometry is read straight
off the shapes the generator drew, which is exactly why mock geometry is reproducible
and why it must never be quoted as model accuracy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.profiles import ACTIVE_PROFILE

# Overlay colours.  Colour is never the only carrier of meaning: every region is also
# numbered and labelled with its class in text, both on the overlay and in the table.
CRACK_RGB = (198, 40, 40)
SCRATCH_RGB = (21, 101, 192)
CLASS_RGB = {"crack": CRACK_RGB, "scratch": SCRATCH_RGB}

# Material -> (base grey, grain strength, texture style, tint)
MATERIAL_STYLE = {
    "steel": (118, 16.0, "brushed", (0, 0, 4)),
    "plastic": (172, 9.0, "matte", (4, 4, 0)),
    "ceramic": (198, 7.0, "speckle", (6, 4, 0)),
    "epoxy": (96, 12.0, "matte", (0, 4, 8)),
    "glass": (150, 6.0, "smooth", (0, 6, 6)),
    "non_steel_metal": (140, 13.0, "brushed", (6, 4, 0)),
}

SOURCE_SIZES = [(640, 480), (800, 600), (720, 540), (960, 720)]


def rng_for(key: str, seed: int) -> np.random.Generator:
    """Deterministic generator for an arbitrary string key."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def make_surface(rng: np.random.Generator, material: str, size: tuple[int, int]) -> Image.Image:
    """Synthesise a plausible industrial surface at the given source resolution."""
    width, height = size
    base, grain, style, tint = MATERIAL_STYLE.get(material, MATERIAL_STYLE["steel"])

    field = rng.normal(base, grain, size=(height, width))

    if style == "brushed":
        # Directional machining marks: strong horizontal correlation.
        lines = rng.normal(0, grain * 0.9, size=(height, 1))
        field += lines
        field += np.sin(np.linspace(0, rng.uniform(30, 60), width))[None, :] * 3.0
    elif style == "speckle":
        speck = rng.random((height, width))
        field += np.where(speck > 0.995, 45, 0)
        field -= np.where(speck < 0.004, 35, 0)
    elif style == "smooth":
        field = field * 0.4 + base * 0.6

    # Uneven illumination - a real station never lights a part evenly.
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height
    radial = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
    field -= radial * rng.uniform(18, 42)

    arr = np.clip(field, 0, 255).astype(np.uint8)
    rgb = np.dstack([
        np.clip(arr.astype(np.int16) + tint[0], 0, 255),
        np.clip(arr.astype(np.int16) + tint[1], 0, 255),
        np.clip(arr.astype(np.int16) + tint[2], 0, 255),
    ]).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    if style in ("matte", "smooth"):
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


@dataclass
class DefectPath:
    class_code: str
    points: list[tuple[float, float]]
    width: float


def make_defect_paths(
    rng: np.random.Generator,
    size: tuple[int, int],
    crack_count: int,
    scratch_count: int,
) -> list[DefectPath]:
    """Build polyline defects.

    Cracks wander and branch-ish with a jagged step; scratches are straighter and
    thinner.  Both are drawn well inside the frame so the bounding box is meaningful.
    """
    width, height = size
    margin = int(min(width, height) * 0.08)
    paths: list[DefectPath] = []

    for kind, count in (("crack", crack_count), ("scratch", scratch_count)):
        for _ in range(count):
            x = rng.uniform(margin, width - margin)
            y = rng.uniform(margin, height - margin)
            angle = rng.uniform(0, 2 * math.pi)
            if kind == "crack":
                steps = int(rng.integers(8, 22))
                step_len = rng.uniform(9, 26)
                wobble = 0.55
                stroke = float(rng.uniform(2.0, 9.0))
            else:
                steps = int(rng.integers(5, 14))
                step_len = rng.uniform(14, 34)
                wobble = 0.10
                stroke = float(rng.uniform(2.0, 12.0))

            pts = [(x, y)]
            for _ in range(steps):
                angle += rng.normal(0, wobble)
                x += math.cos(angle) * step_len
                y += math.sin(angle) * step_len
                x = float(np.clip(x, margin * 0.5, width - margin * 0.5))
                y = float(np.clip(y, margin * 0.5, height - margin * 0.5))
                pts.append((x, y))
            paths.append(DefectPath(kind, pts, stroke))

    return paths


def _path_mask(path: DefectPath, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    w = max(1, int(round(path.width)))
    draw.line(path.points, fill=255, width=w, joint="curve")
    # Round the caps so the inscribed circle at the ends matches the stroke width.
    r = w / 2.0
    for px, py in (path.points[0], path.points[-1]):
        draw.ellipse([px - r, py - r, px + r, py + r], fill=255)
    return mask


def polyline_length(points: list[tuple[float, float]]) -> float:
    """Arc length along the centreline.

    Summing pixels instead of arc length would understate a 45-degree defect by about
    41 % - the same reason the real pipeline counts a diagonal skeleton step as sqrt(2).
    """
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


@dataclass
class MockRegion:
    class_code: str
    area_px: int
    length_px: float
    max_width_px: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mask: Image.Image


def measure_paths(paths: list[DefectPath], size: tuple[int, int]) -> list[MockRegion]:
    """Rasterise each path and measure it, applying the profile's region floors.

    The floors come from ``app/postprocess.py::Profile`` - the same object the real
    pipeline uses - so a re-tuned floor changes the mock data too and the two can
    never disagree about what counts as a region.
    """
    regions: list[MockRegion] = []
    for path in paths:
        mask = _path_mask(path, size)
        arr = np.array(mask)
        ys, xs = np.nonzero(arr)
        if xs.size == 0:
            continue

        area = int(xs.size)
        length = polyline_length(path.points)
        if area < ACTIVE_PROFILE.min_area_px or length < ACTIVE_PROFILE.min_skeleton_px:
            continue  # below the clean-surface false-positive floors

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        regions.append(
            MockRegion(
                class_code=path.class_code,
                area_px=area,
                length_px=length,
                max_width_px=float(path.width),
                bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                centroid=(float(xs.mean()), float(ys.mean())),
                mask=mask,
            )
        )
    return regions


def burn_defects(surface: Image.Image, regions: list[MockRegion]) -> Image.Image:
    """Darken the surface where a defect exists, so the source image shows the flaw."""
    out = surface.copy()
    arr = np.array(out).astype(np.int16)
    for region in regions:
        m = np.array(region.mask) > 0
        arr[m] = np.clip(arr[m] * 0.35, 0, 255)
        edge = np.array(region.mask.filter(ImageFilter.MaxFilter(3))) > 0
        halo = edge & ~m
        arr[halo] = np.clip(arr[halo] * 0.8, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def draw_overlay(source: Image.Image, regions: list[MockRegion]) -> Image.Image:
    """Draw masks, numbered boxes and class labels at SOURCE resolution.

    Source resolution, not 256x256, so the overlay registers to the actual part (T-13).
    """
    overlay = source.convert("RGB")
    tint = overlay.copy()

    for region in regions:
        colour = CLASS_RGB.get(region.class_code, CRACK_RGB)
        solid = Image.new("RGB", overlay.size, colour)
        tint = Image.composite(solid, tint, region.mask)

    overlay = Image.blend(overlay, tint, 0.72)
    draw = ImageDraw.Draw(overlay)
    label_font = _font(max(12, overlay.width // 55))

    for index, region in enumerate(regions, start=1):
        colour = CLASS_RGB.get(region.class_code, CRACK_RGB)
        x, y, w, h = region.bbox
        pad = 6
        box = [x - pad, y - pad, x + w + pad, y + h + pad]
        draw.rectangle(box, outline=colour, width=2)

        # Text label - the overlay never relies on colour alone.
        text = f"{index} {region.class_code}"
        tx, ty = box[0], max(0, box[1] - (label_font.size + 6))
        try:
            tw = int(draw.textlength(text, font=label_font))
        except AttributeError:
            tw = len(text) * 7
        draw.rectangle([tx, ty, tx + tw + 8, ty + label_font.size + 4], fill=colour)
        draw.text((tx + 4, ty + 2), text, fill=(255, 255, 255), font=label_font)

    return overlay


def crop_region(image: Image.Image, bbox: tuple[int, int, int, int], zoom: int = 3) -> Image.Image:
    """Crop around a region with context, then enlarge - the region-detail centre pane."""
    x, y, w, h = bbox
    pad = int(max(w, h) * 0.6) + 24
    box = (
        max(0, x - pad),
        max(0, y - pad),
        min(image.width, x + w + pad),
        min(image.height, y + h + pad),
    )
    crop = image.crop(box)
    if crop.width == 0 or crop.height == 0:
        return image
    return crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)


def unreadable_placeholder(size: tuple[int, int] = (640, 480)) -> Image.Image:
    """A hatched panel used where an image could not be read.

    Hatching rather than a colour fill, so the state survives glare and colour blindness.
    """
    img = Image.new("RGB", size, (232, 232, 232))
    draw = ImageDraw.Draw(img)
    for offset in range(-size[1], size[0], 14):
        draw.line([(offset, 0), (offset + size[1], size[1])], fill=(196, 196, 196), width=3)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(90, 90, 90), width=3)
    font = _font(22)
    draw.text((18, size[1] // 2 - 14), "IMAGE COULD NOT BE READ", fill=(40, 40, 40), font=font)
    return img
