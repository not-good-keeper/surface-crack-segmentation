"""Composite crack geometry onto a real background.

Everything here exists to remove a shortcut. A tiny segmentation model will
happily learn "crack = the region with paste artifacts" instead of "crack = thin
dark branching groove", and if it does, it scores beautifully on synthetic data
and fails on the first real photograph. The specific defences:

  multiplicative darkening   a crack darkens what is underneath it; it does not
                             replace it. Additive/flat-fill lines destroy the
                             substrate texture and are trivially separable.
  groove shading             real cracks have depth: dark core, soft shoulder,
                             and a bright lip on the lit side.
  blur matching             *the* giveaway. A razor-sharp crack on a slightly
                             soft photo is detectable by a 3x3 filter. We measure
                             background sharpness and blur the crack to match.
  noise matching             crack interiors must carry the same sensor noise as
                             their surroundings.
  supersampling              render at 2x and box-down, so edges are anti-aliased
                             the way an optical system produces them.
  honest masks               a crack that fades below the local noise floor is not
                             visible, so it is not labelled. Otherwise we train the
                             model to hallucinate.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class RenderParams:
    darkness: float = 0.62        # peak multiplicative darkening at the core
    shoulder: float = 1.6         # px, softness of the groove wall
    lip: float = 0.22             # brightness of the lit lip, 0 disables
    light_angle: float = 0.9      # rad, direction the light comes from
    noise_gain: float = 1.0       # how much local noise to re-inject
    blur_match: float = 1.0       # 0..1, how strictly to match background blur
    contrast_floor: float = 0.055 # mask honesty threshold (fraction of dynamic range)
    supersample: int = 2
    dirt_prob: float = 0.25       # occasional occlusion by dirt/shadow
    fade_prob: float = 0.35       # fade one end below visibility


def rasterize(polys, size, ss: int = 2):
    """Draw polylines into a float32 'depth' field in [0,1] at ss x resolution."""
    h, w = size
    canvas = np.zeros((h * ss, w * ss), np.float32)
    for pts, wts in polys:
        p = (pts * ss).astype(np.int32)
        for i in range(len(p) - 1):
            r = max(1, int(round(wts[i] * ss)))
            cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), 1.0, r, cv2.LINE_AA)
    return canvas


def estimate_sharpness(gray):
    """Blur sigma the background appears to carry. Higher variance => sharper."""
    v = cv2.Laplacian(gray, cv2.CV_32F).var()
    # empirical map: crisp photos land ~300+, soft ones ~20
    return float(np.clip(1.9 - 0.55 * np.log1p(v), 0.25, 2.6))


def estimate_noise(gray):
    """Robust per-pixel noise sigma via median-absolute-deviation of a high-pass."""
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def render(bg: np.ndarray, polys, rp: RenderParams,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (composited BGR uint8, binary mask uint8) at bg resolution."""
    h, w = bg.shape[:2]
    ss = max(1, rp.supersample)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    # ---- geometry -> soft groove profile ---------------------------------
    d = rasterize(polys, (h, w), ss)
    if ss > 1:
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
    if d.max() <= 1e-6:
        return bg.copy(), np.zeros((h, w), np.uint8)

    core = np.clip(d, 0, 1)
    core_geo = core.copy()          # geometric extent, BEFORE optical blur.
                                    # The mask is built from this: optical blur
                                    # models the camera, it does not widen the
                                    # crack, and thresholding a blurred 1 px line
                                    # fragments it into dashes.
    shoulder = cv2.GaussianBlur(core, (0, 0), max(0.35, rp.shoulder))
    profile = np.clip(core + 0.55 * shoulder, 0, 1)

    # ---- match the background's optical blur ------------------------------
    sigma = estimate_sharpness(gray) * rp.blur_match
    if sigma > 0.3:
        profile = cv2.GaussianBlur(profile, (0, 0), sigma)
        core = cv2.GaussianBlur(core, (0, 0), sigma * 0.7)

    # ---- fade one end / occlude, so not every crack is fully visible -------
    vis = np.ones_like(profile)
    if rng.random() < rp.fade_prob:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, 2 * np.pi)
        t = (xx * np.cos(a) + yy * np.sin(a))
        t = (t - t.min()) / max(float(np.ptp(t)), 1e-6)
        vis *= np.clip(rng.uniform(0.15, 0.75) + t, 0, 1)
    if rng.random() < rp.dirt_prob:
        blob = np.zeros((h, w), np.float32)
        for _ in range(rng.integers(1, 4)):
            cv2.circle(blob, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                       int(rng.uniform(0.05, 0.18) * max(h, w)), 1.0, -1)
        vis *= 1.0 - 0.85 * cv2.GaussianBlur(blob, (0, 0), 9)

    profile = profile * vis
    core = core * vis
    core_geo = core_geo * vis

    # ---- multiplicative darkening (respects substrate texture) ------------
    img = bg.astype(np.float32)
    atten = 1.0 - rp.darkness * profile[..., None]
    out = img * atten

    # ---- bright lip on the lit side (groove has depth) --------------------
    if rp.lip > 0:
        gx = cv2.Sobel(profile, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(profile, cv2.CV_32F, 0, 1, ksize=3)
        lit = np.clip(gx * np.cos(rp.light_angle) + gy * np.sin(rp.light_angle), 0, None)
        lit = cv2.GaussianBlur(lit, (0, 0), 1.0)
        if lit.max() > 1e-6:
            lit /= lit.max()
        out += (rp.lip * 90.0) * lit[..., None]

    # ---- re-inject the background's own noise into the crack --------------
    ns = estimate_noise(gray) * rp.noise_gain
    if ns > 0.2:
        out += rng.normal(0, ns, out.shape).astype(np.float32) * profile[..., None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # ---- honest mask ------------------------------------------------------
    # Label only what is actually visible: geometry AND enough local contrast to
    # be distinguishable from the substrate.
    delta = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32) - \
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_range = float(np.percentile(gray, 95) - np.percentile(gray, 5)) + 1e-6
    # Visibility is judged on a lightly smoothed delta so single-pixel noise does
    # not punch holes along an otherwise clearly visible crack.
    visible = cv2.GaussianBlur(delta, (0, 0), 0.8) > max(1.5, rp.contrast_floor * local_range)
    mask = ((core_geo > 0.35) & visible).astype(np.uint8) * 255
    # Close 1 px gaps left where the crack crosses a locally dark patch: the crack
    # is continuous in reality, and a dashed label teaches broken predictions.
    if mask.any():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    return out, mask


def random_params(rng, material="mixed") -> RenderParams:
    rp = RenderParams(
        darkness=float(rng.uniform(0.30, 0.80)),
        shoulder=float(rng.uniform(0.8, 2.6)),
        lip=float(rng.uniform(0.0, 0.35)),
        light_angle=float(rng.uniform(0, 2 * np.pi)),
        noise_gain=float(rng.uniform(0.6, 1.5)),
        blur_match=float(rng.uniform(0.75, 1.25)),
    )
    if material in ("steel", "metal"):
        rp.darkness *= 0.8   # cracks on bright metal are lower contrast
        rp.lip = max(rp.lip, 0.18)   # specular lip is pronounced
    if material == "ceramic":
        rp.darkness *= 0.72  # glaze crazing is subtle
        rp.shoulder *= 0.7
    if material == "wood":
        rp.darkness *= 1.05  # checks are dark and open
    return rp
