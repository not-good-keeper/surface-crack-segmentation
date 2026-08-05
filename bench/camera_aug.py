"""Camera-pipeline augmentation, applied at training time.

Adapted from the camera simulation in the companion DefectForge engine, which models
what happens between a surface and a JPEG on a phone: optics, sensor, and ISP. Our own
synthetic generator composites cracks onto real photographs and then stops, so the
substrate carries whatever camera signature the original photographer had and the crack
carries none. A model can learn that mismatch instead of learning crack morphology.

Doing this as a training-time augmentation rather than baking it into generation is a
deliberate choice: it costs no regeneration, it can be ablated on and off against a
frozen dataset, and it multiplies every existing sample instead of fixing each one once.

Split matters:
  geometric (perspective, rotation)  -> image AND mask, mask with nearest neighbour
  photometric (blur, noise, JPEG...) -> image ONLY

Applying a photometric effect to a mask would be meaningless; applying a geometric one
to only the image would silently decouple label from pixel, which is the failure this
whole project keeps guarding against.
"""
from __future__ import annotations

import cv2
import numpy as np


def _motion_blur(img, rng):
    k = int(rng.integers(3, 10))
    ang = float(rng.uniform(0, 180))
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    s = ker.sum()
    if s <= 1e-6:
        return img
    return cv2.filter2D(img, -1, ker / s)


def _defocus(img, rng):
    r = int(rng.integers(2, 5))
    ker = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(ker, (r, r), r, 1.0, -1)
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def _chromatic_aberration(img, rng):
    """Scale the red and blue planes slightly differently about the centre."""
    s = float(rng.uniform(0.0015, 0.006))
    h, w = img.shape[:2]
    out = img.copy()
    for c, sign in ((0, -1.0), (2, 1.0)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 1.0 + sign * s)
        out[..., c] = cv2.warpAffine(img[..., c], M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)
    return out


def _vignette(img, rng):
    h, w = img.shape[:2]
    strength = float(rng.uniform(0.12, 0.42))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2)
    return img * (1.0 - strength * r[..., None] ** 2)


def _ring_light(img, rng):
    """Approximate a centred borescope ring-light hot spot.

    This is deliberately photometric: it changes illumination, not scene geometry.
    A broad centre lift plus edge fall-off is a closer model of a snake-camera LED
    ring than a generic brightness jitter, and leaves the target mask untouched.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) * float(rng.uniform(0.43, 0.57))
    cx = (w - 1) * float(rng.uniform(0.43, 0.57))
    scale = float(max(h, w))
    r2 = ((yy - cy) / scale) ** 2 + ((xx - cx) / scale) ** 2
    gain = 1.0 + float(rng.uniform(0.12, 0.38)) * np.exp(
        -r2 / float(rng.uniform(0.045, 0.12)))
    return img * gain[..., None]


def _barrel_distortion(img, msk, rng):
    """Apply a fisheye-like barrel warp consistently to image and mask."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # Map destination coordinates to source coordinates.  The positive radial
    # term creates the characteristic edge stretching of a close borescope view.
    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    k = float(rng.uniform(0.08, 0.28))
    scale = 1.0 + k * (xn * xn + yn * yn)
    map_x = (cx + xn * scale * max(cx, 1.0)).astype(np.float32)
    map_y = (cy + yn * scale * max(cy, 1.0)).astype(np.float32)
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    msk = cv2.remap(msk, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, msk


def _rolling_shutter_shear(img, msk, rng):
    """Row-dependent horizontal shear, applied consistently to labels."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = float(rng.uniform(-0.045, 0.045)) * w
    # A linear row-time model avoids an implausible wavy warp while still
    # representing movement during scanout.
    shift = amp * (yy / max(h - 1, 1) - 0.5)
    map_x = (xx - shift).astype(np.float32)
    map_y = yy.astype(np.float32)
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    msk = cv2.remap(msk, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, msk


def _sensor_noise(img, rng):
    """Shot noise scales with signal; read noise does not."""
    iso = float(rng.uniform(1.5, 9.0))
    sig = np.clip(img, 0, 255)
    shot = rng.normal(0.0, 1.0, img.shape).astype(np.float32) * np.sqrt(sig / 255.0) * iso
    read = rng.normal(0.0, iso * 0.45, img.shape).astype(np.float32)
    return img + shot + read


def _jpeg(img, rng):
    q = int(rng.integers(45, 92))
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)


def _belt_motion_blur(img, rng, axis_deg=0.0):
    """Motion blur along the belt axis.

    A handheld photograph is smeared in whatever direction the hand moved, so the
    generic `_motion_blur` randomises the angle over 180 degrees. A part on a conveyor
    moves along one fixed axis under a fixed camera, so the smear direction is a
    property of the installation and barely varies. Randomising it would teach the
    model that defect orientation and blur orientation are independent, which on a
    real line they are not -- a scratch parallel to travel and one across it degrade
    differently, and that asymmetry is exactly what the model has to survive.
    """
    k = int(rng.integers(3, 8))
    ang = axis_deg + float(rng.uniform(-6, 6))
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    s = ker.sum()
    if s <= 1e-6:
        return img
    return cv2.filter2D(img, -1, ker / s)


def _specular_highlight(img, rng):
    """Blown-out reflection of the line's LED bar off a polished surface.

    This is the dominant artefact on the material we prioritise. Steel, chrome and
    glazed ceramic under controlled illumination produce elongated specular streaks
    that are bright, thin and directional -- the same description as a scratch. A
    model that has never seen one classifies it as a defect, which is the
    over-rejection failure NFR-3 exists to prevent. Photometric only: it adds light,
    it does not add a defect, so the mask must NOT follow it.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) * float(rng.uniform(0.15, 0.85))
    cx = (w - 1) * float(rng.uniform(0.15, 0.85))
    ang = np.deg2rad(float(rng.uniform(0, 180)))
    # rotate into the streak frame, then use very different sigmas along/across
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    su = float(rng.uniform(0.10, 0.40)) * max(h, w)
    sv = su * float(rng.uniform(0.04, 0.18))
    streak = np.exp(-(u ** 2) / (2 * su ** 2) - (v ** 2) / (2 * sv ** 2))
    return img + float(rng.uniform(40, 165)) * streak[..., None]


def geometric(img, msk, rng, strength=1.0, profile="handheld"):
    """Perspective, fisheye and shear applied consistently to image and mask."""
    h, w = img.shape[:2]
    if profile == "conveyor":
        # A fixed camera at a fixed working distance over a flat belt sees almost no
        # perspective change and almost no scale change between parts. What does vary
        # is where the part sits on the belt and how it is rotated in-plane.
        ang = float(rng.uniform(-180, 180))
        zoom = float(rng.uniform(0.97, 1.04))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, zoom)
        M[0, 2] += float(rng.uniform(-0.05, 0.05)) * w
        M[1, 2] += float(rng.uniform(-0.05, 0.05)) * h
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # No barrel warp: a machine-vision lens on an inspection station is chosen and
        # calibrated to be rectilinear, so simulating fisheye here would train the
        # model to undo a distortion the deployment optics do not have.
        if rng.random() < 0.20:
            img, msk = _rolling_shutter_shear(img, msk, rng)
        return img, msk

    ang = float(rng.uniform(-8, 8)) * strength
    zoom = float(rng.uniform(0.92, 1.10))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, zoom)
    d = float(rng.uniform(0.0, 0.045)) * strength * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    A = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    H = P @ A
    img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    # nearest for labels, and a zero border: reflecting a label outward would invent
    # annotation with no image evidence behind it
    msk = cv2.warpPerspective(msk, H, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.35:
        img, msk = _barrel_distortion(img, msk, rng)
    if rng.random() < 0.25:
        img, msk = _rolling_shutter_shear(img, msk, rng)
    return img, msk


def photometric(img_u8, rng, p=0.85, profile="handheld"):
    """Optics -> sensor -> ISP, in that physical order. Image only."""
    if rng.random() > p:
        return img_u8
    img = img_u8.astype(np.float32)

    if profile == "conveyor":
        # Controlled illumination and a fixed focus remove most of the handheld
        # variation, so this profile is deliberately narrower than the handheld one --
        # and spends the budget it saves on the two artefacts that actually dominate a
        # line: belt-axis smear and specular glare off polished product.
        if rng.random() < 0.45:
            img = _belt_motion_blur(img, rng)
        elif rng.random() < 0.10:
            img = _defocus(img, rng)          # occasional focus drift / part height
        if rng.random() < 0.55:
            img = _specular_highlight(img, rng)
        if rng.random() < 0.35:
            img = _ring_light(img, rng)       # overhead LED falloff
        if rng.random() < 0.25:
            img = _vignette(img, rng)
        if rng.random() < 0.70:
            img = _sensor_noise(img, rng)
        if rng.random() < 0.45:
            img = _jpeg(img, rng)
        return np.clip(img, 0, 255).astype(np.uint8)

    # At most ONE blur. Stacking motion blur on defocus on heavy noise can erase a
    # 2 px crack outright while the mask still asserts it, which trains the model to
    # hallucinate cracks from texture. A real photo is blurred OR defocused, rarely
    # both severely.
    u = rng.random()
    if u < 0.32:
        img = _motion_blur(img, rng)
    elif u < 0.58:
        img = _defocus(img, rng)
    if rng.random() < 0.30:
        img = _chromatic_aberration(img, rng)
    if rng.random() < 0.45:
        img = _vignette(img, rng)
    if rng.random() < 0.30:
        img = _ring_light(img, rng)
    if rng.random() < 0.75:
        img = _sensor_noise(img, rng)
    if rng.random() < 0.60:
        img = _jpeg(img, rng)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply(img_u8, msk_u8, rng, geo_p=0.5, photo_p=0.85, profile="handheld"):
    """`profile` selects the capture model.

    `handheld` is the original phone/borescope model and is kept unchanged so every
    stored benchmark remains reproducible. `conveyor` models the deployment target in
    docs/ARCHITECTURE.md section 5: a fixed overhead camera, constant working distance,
    controlled illumination, part moving along one axis.
    """
    if profile not in ("handheld", "conveyor"):
        raise ValueError(f"unknown camera profile {profile!r}")
    if rng.random() < geo_p:
        f_img, f_msk = geometric(img_u8.astype(np.float32), msk_u8, rng,
                                 profile=profile)
        img_u8 = np.clip(f_img, 0, 255).astype(np.uint8)
        msk_u8 = f_msk
    return photometric(img_u8, rng, photo_p, profile=profile), msk_u8
