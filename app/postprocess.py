"""Segmentation logits -> defect regions with geometry (ARCHITECTURE.md section 7).

Shared by the UI and the batch CLI so an on-screen number and a QC report cannot drift.
Deterministic: same logits + same profile -> same regions.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import cv2
import numpy as np

BACKGROUND, CRACK, SCRATCH = 0, 1, 2
CLASS_NAMES = {CRACK: "crack", SCRATCH: "scratch"}
CLASS_COLOUR = {CRACK: (60, 60, 235), SCRATCH: (235, 170, 40)}    # BGR


@dataclass(frozen=True)
class Profile:
    """Versioned post-processing config, recorded with every result.

    Thresholds are per class (a scratch is wider and higher-contrast than a hairline
    crack) and were chosen on val by bench/class_thresh.py, never on a test split. The
    area and skeleton floors come from the false-positive budget, NFR-3.
    """

    profile_id: str = "conveyor-v2"
    crack_thresh: float = 0.40
    scratch_thresh: float = 0.20
    min_area_px: int = 24
    min_skeleton_px: int = 6
    connectivity: int = 8


def softmax(logits):
    """(C,H,W) logits -> probabilities, shifted for numerical stability."""
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def class_map(prob, profile):
    """Probabilities -> class indices. Takes (C,H,W) or batched (N,C,H,W).

    Below both floors is background; above both, the higher score wins. The only
    implementation of section 7.1 -- the evaluation scripts call this, so the rule the
    benchmark measures cannot diverge from the rule the app ships.
    """
    t = np.array([0.0, profile.crack_thresh, profile.scratch_thresh])[:prob.shape[-3]]
    fg = np.moveaxis(prob, -3, 0)[1:]
    passes = fg >= t[1:].reshape((-1,) + (1,) * (fg.ndim - 1))
    best = np.where(passes, fg, -1.0).argmax(axis=0) + 1
    return np.where(passes.any(axis=0), best, BACKGROUND).astype(np.int32)


def _skeleton(mask):
    """Medial axis plus arc length, diagonal steps counted as sqrt(2).

    Summing skeleton pixels would understate a diagonal defect by ~41 %.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(mask.astype(bool))
    if not skel.any():
        return skel, 0.0
    s = skel.astype(np.uint8)
    ortho = int((s[:, :-1] & s[:, 1:]).sum()) + int((s[:-1, :] & s[1:, :]).sum())
    diag = int((s[:-1, :-1] & s[1:, 1:]).sum()) + int((s[:-1, 1:] & s[1:, :-1]).sum())
    # An isolated pixel has no links; report its own extent rather than zero.
    return skel, (ortho + diag * np.sqrt(2.0)) or 1.0


def _max_width(mask, skel):
    """Twice the largest inscribed-circle radius along the centreline.

    Not area/length, which averages away a local bulge. Note this is a lower bound on
    perpendicular extent: an inscribed circle is bounded by the smallest local
    dimension, so it under-reports chip-shaped regions (out of scope, section 4).
    """
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2.0 * dt[skel].max()) if skel.any() else 0.0


def extract_regions(cmap, profile):
    """Connected components per class -> list of region dicts.

    Per class rather than over the union: two touching defects of different types are
    two findings, and merging them would report one region with a nonsense type.
    """
    regions = []
    for cls, cname in CLASS_NAMES.items():
        binary = (cmap == cls).astype(np.uint8)
        if not binary.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=profile.connectivity)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < profile.min_area_px:
                continue
            comp = labels == i
            skel, length = _skeleton(comp)
            if skel.sum() < profile.min_skeleton_px:
                continue
            regions.append({
                "id": len(regions) + 1,
                "type": cname,
                "area_px": area,
                "length_px": round(length, 2),
                "max_width_px": round(_max_width(comp, skel), 2),
                "bbox_xywh": [int(stats[i, c]) for c in
                              (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                               cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)],
            })
    return regions


def overlay(image_bgr, cmap, regions, alpha=0.45):
    """Semi-transparent class masks plus region ids, in source-image coordinates."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    sx, sy = w / max(cmap.shape[1], 1), h / max(cmap.shape[0], 1)
    if cmap.shape[:2] != (h, w):
        cmap = cv2.resize(cmap.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(np.int32)

    tint = out.copy()
    for cls, colour in CLASS_COLOUR.items():
        tint[cmap == cls] = colour
    out = cv2.addWeighted(tint, alpha, out, 1 - alpha, 0)

    for r in regions:
        x, y, bw, bh = r["bbox_xywh"]
        x, y, bw, bh = int(x * sx), int(y * sy), int(bw * sx), int(bh * sy)
        colour = CLASS_COLOUR[CRACK if r["type"] == "crack" else SCRATCH]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 1)
        cv2.putText(out, f"{r['id']}:{r['type'][:3]}", (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return out


def build_record(image_bytes, regions, profile, model_version, **meta):
    """The ARCHITECTURE.md section 8.1 record. `meta` carries station_id, product_id,
    material and overlay_path.

    `empty` is written explicitly rather than inferred from an empty list: a clean
    inspection and a capture that never produced a result must never look the same in
    a QC history.
    """
    return {
        "schema_version": "1.0",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "model_version": model_version,
        "station_id": meta.get("station_id", "unset"),
        "product_id": meta.get("product_id"),
        "material": meta.get("material"),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processing_profile": profile.profile_id,
        "profile": asdict(profile),
        "regions": regions,
        "empty": not regions,
        "overlay_path": meta.get("overlay_path"),
    }


def record_to_rows(record):
    """Flatten one record into CSV rows -- one per region, one for a clean product.

    A defect-free product still gets a row: a report where clean items are simply absent
    cannot be distinguished from one where they were never inspected.
    """
    head = {k: record[k] for k in ("image_sha256", "model_version", "station_id",
                                   "product_id", "processed_at", "processing_profile")}
    if not record["regions"]:
        return [dict(head, region_id="", type="none", area_px=0, length_px=0.0,
                     max_width_px=0.0, bbox_xywh="")]
    return [dict(head, region_id=r["id"], type=r["type"], area_px=r["area_px"],
                 length_px=r["length_px"], max_width_px=r["max_width_px"],
                 bbox_xywh=json.dumps(r["bbox_xywh"])) for r in record["regions"]]
