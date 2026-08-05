"""Procedural crack geometry.

A crack is represented as a set of polylines with a per-vertex half-width, which
is what both the renderer and the mask generator consume.

Growth model: a correlated random walk. Direction persists and drifts by a small
random turn each step rather than being re-drawn, because independent per-step
directions produce a wandering scribble whose curvature statistics look nothing
like a real crack. Width tapers toward the tips -- real cracks are widest at the
origin and close to zero where they arrest -- and branches inherit a fraction of
the parent width, which is what stops branch tips from looking like main trunks.

All randomness flows through an explicit numpy Generator so every crack is
reproducible from (seed, params) alone. That is what makes the eventual prune of
the bulk pixels safe: nothing here depends on global RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

Material = Literal["concrete", "asphalt", "masonry", "ceramic", "steel",
                   "metal", "plastic", "wood", "mixed"]


@dataclass
class CrackParams:
    """Growth parameters. Defaults describe a mid-scale concrete crack."""
    n_steps: int = 140              # trunk length in steps
    step: float = 3.0               # px per step at render resolution
    turn_sigma: float = 0.20        # rad; per-step direction jitter
    persistence: float = 0.82       # 0..1, how strongly heading is retained
    width0: float = 9.0             # half-width at the origin, px
    width_taper: float = 0.85       # width multiplier along the trunk
    branch_prob: float = 0.012      # per-step chance of spawning a branch
    branch_angle: float = 0.6       # rad, mean deviation of a branch
    branch_angle_sigma: float = 0.25
    branch_width_frac: float = 0.55
    max_depth: int = 3
    min_width: float = 0.35
    wander: float = 0.0             # low-frequency drift, adds large-scale curvature
    # Material-specific behaviours
    align_to: float | None = None   # radians; e.g. wood grain direction
    align_strength: float = 0.0     # 0..1 pull toward align_to
    network: bool = False           # ceramic crazing: closed polygonal network
    radial: bool = False            # plastic: star fracture from an impact point


MATERIAL_PRESETS: dict[str, dict] = {
    # Wide, heavily branched, high curvature.
    "concrete": dict(turn_sigma=0.22, branch_prob=0.014, width0=9.0, n_steps=150,
                     persistence=0.80, wander=0.010),
    "asphalt": dict(turn_sigma=0.26, branch_prob=0.020, width0=18.0, n_steps=170,
                    persistence=0.76, wander=0.014),
    # Mortar joints dominate: straighter runs, sharp direction changes.
    "masonry": dict(turn_sigma=0.14, branch_prob=0.008, width0=24.0, n_steps=120,
                    persistence=0.88, wander=0.004),
    # Glaze crazing: fine, dense, closed polygonal network.
    # Glaze crazing: fine and interconnected, but the earlier settings produced a
    # denser web than is ever actually visible, and labelling all of it taught the
    # model to over-segment. Fewer seeds, lower branch rate, shorter runs.
    # 8.4 measured 1.37x too thick against real ceramic (2.14% vs 1.56% fg).
    "ceramic": dict(turn_sigma=0.26, branch_prob=0.022, width0=6.1, n_steps=60,
                    persistence=0.74, max_depth=3, network=True, wander=0.0),
    # Fatigue cracks: fine, straight, minimal branching, follow stress lines.
    "steel": dict(turn_sigma=0.07, branch_prob=0.003, width0=4.6, n_steps=130,
                  persistence=0.94, wander=0.002),
    "metal": dict(turn_sigma=0.09, branch_prob=0.004, width0=5.2, n_steps=120,
                  persistence=0.92, wander=0.003),
    # Impact fracture: radial star. Width measured against KolektorSDD1,
    # whose real plastic cracks are microscopic fractures in the commutator
    # embedding (0.50% fg) -- the first setting was 6.8x too thick.
    "plastic": dict(turn_sigma=0.10, branch_prob=0.006, width0=1.3, n_steps=80,
                    persistence=0.91, radial=True, width_taper=0.72),
    # Checks and shakes run ALONG the grain.
    "wood": dict(turn_sigma=0.08, branch_prob=0.005, width0=8.4, n_steps=160,
                 persistence=0.93, align_strength=0.75, wander=0.003),
    "mixed": dict(),
}


def params_for(material, rng, scale=1.0) -> CrackParams:
    """Preset for a material, with per-sample jitter so no two cracks match.

    `scale` models camera distance: it stretches both length and width together,
    because apparent crack width in pixels is a function of how close the phone is.
    """
    p = CrackParams(**MATERIAL_PRESETS.get(material, {}))
    j = lambda v, f: float(v * rng.uniform(1 - f, 1 + f))  # noqa: E731
    p.n_steps = max(12, int(j(p.n_steps, 0.45) * scale ** 0.5))
    p.step = j(p.step, 0.25)
    p.turn_sigma = j(p.turn_sigma, 0.40)
    p.persistence = float(np.clip(j(p.persistence, 0.06), 0.5, 0.985))
    p.width0 = max(0.5, j(p.width0, 0.50) * scale)
    p.branch_prob = j(p.branch_prob, 0.55)
    p.wander = j(p.wander, 0.6)
    return p


def _grow(rng, x, y, ang, width, p, depth, size, out):
    """One branch of the walk; appends (points, widths) to `out`."""
    h, w = size
    pts, wts = [(x, y)], [width]
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(p.n_steps):
        # correlated turn: retain heading, add jitter, plus slow global drift
        turn = rng.normal(0, p.turn_sigma)
        ang = p.persistence * ang + (1 - p.persistence) * (ang + turn) + turn * 0.5
        if p.wander:
            ang += p.wander * np.sin(phase + i * 0.08)
        if p.align_strength and p.align_to is not None:
            # pull heading toward the grain direction (mod pi: grain is undirected)
            d = np.arctan2(np.sin(p.align_to - ang), np.cos(p.align_to - ang))
            if abs(d) > np.pi / 2:
                d = d - np.sign(d) * np.pi
            ang += p.align_strength * 0.25 * d

        x += p.step * np.cos(ang)
        y += p.step * np.sin(ang)
        width *= p.width_taper ** (1.0 / max(p.n_steps, 1))
        width = max(width, p.min_width)
        if not (-20 <= x < w + 20 and -20 <= y < h + 20):
            break                                  # ran off frame: fine, realistic
        pts.append((x, y))
        wts.append(width)

        if depth < p.max_depth and rng.random() < p.branch_prob:
            sign = 1 if rng.random() < 0.5 else -1
            ba = ang + sign * abs(rng.normal(p.branch_angle, p.branch_angle_sigma))
            child = CrackParams(**{**asdict(p),
                                   "n_steps": max(8, int(p.n_steps * rng.uniform(0.25, 0.6))),
                                   "branch_prob": p.branch_prob * 0.6})
            _grow(rng, x, y, ba, width * p.branch_width_frac, child,
                  depth + 1, size, out)

    if len(pts) > 2:
        out.append((np.asarray(pts, np.float32), np.asarray(wts, np.float32)))


def generate(size, material, rng, scale=1.0, grain_angle=None):
    """Return [(points[N,2], half_widths[N]), ...] for one crack instance."""
    h, w = size
    p = params_for(material, rng, scale)
    if grain_angle is not None:
        p.align_to = grain_angle

    out: list = []

    if p.radial:
        # Impact star: several arms from a common origin.
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        for k in range(rng.integers(3, 7)):
            a = 2 * np.pi * k / 5 + rng.normal(0, 0.4)
            _grow(rng, cx, cy, a, p.width0, p, 1, size, out)
        return out

    if p.network:
        # Glaze crazing: several seeds, high branch rate, short runs -> closes into
        # a polygonal network rather than one long line.
        for _ in range(rng.integers(2, 4)):
            sx, sy = rng.uniform(0, w), rng.uniform(0, h)
            _grow(rng, sx, sy, rng.uniform(0, 2 * np.pi), p.width0, p, 0, size, out)
        return out

    # Default: enter from an edge so the crack crosses the frame like a real one.
    edge = rng.integers(0, 4)
    if edge == 0:
        sx, sy, a = rng.uniform(0, w), 0.0, rng.uniform(0.2, np.pi - 0.2)
    elif edge == 1:
        sx, sy, a = rng.uniform(0, w), float(h), rng.uniform(-np.pi + 0.2, -0.2)
    elif edge == 2:
        sx, sy, a = 0.0, rng.uniform(0, h), rng.uniform(-np.pi / 2 + 0.2, np.pi / 2 - 0.2)
    else:
        sx, sy, a = float(w), rng.uniform(0, h), rng.uniform(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2)
    _grow(rng, sx, sy, a, p.width0, p, 0, size, out)
    return out
