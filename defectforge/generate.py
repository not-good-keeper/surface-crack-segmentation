"""DefectForge generation: synthetic cracks and synthetic hard negatives.

Every sample is fully described by its provenance row, so the pixels can be
deleted and rebuilt byte-identically later:

    patch_id, generator_version, rng_seed, bg_id, material, kind, params...

Determinism comes from one Generator per sample, seeded from (run_seed, index).
Nothing reads global RNG state, so generating sample 91,234 alone reproduces
exactly the sample that a full run would have produced at that index.

Two kinds are produced:
  kind='crack'      labelled foreground
  kind='negative'   crack-LIKE structures labelled BACKGROUND: scratches, mould
                    lines, weld seams, grain, grout and cable shadows. These cost
                    almost nothing to render and are the main defence against an
                    inspection app that flags every dark line it sees.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import generate as gen_crack  # noqa: E402
from render import RenderParams, render, random_params  # noqa: E402

GENERATOR_VERSION = "df-1.0.0"   # bump on ANY change to geometry/render output

ROOT = Path(__file__).resolve().parent.parent
BG_CSV = ROOT / "data/backgrounds.csv"
OUT = ROOT / "data/synth"
SIZE = 256          # final patch size
RENDER_SIZE = 512   # render above target, downsample -> optical-style anti-aliasing


# --------------------------------------------------------------- negatives
def draw_negative(bg, rng, material):
    """Crack-like clutter, labelled background.

    Each of these is something a naive crack model reliably fires on. They are
    rendered with the SAME pipeline as cracks so the model cannot separate them
    by rendering artifacts -- only by the structure itself, which is the point.
    """
    h, w = bg.shape[:2]
    img = bg.astype(np.float32)
    kind = rng.choice(["scratch", "seam", "grain", "grout", "cable", "spatter"])
    layer = np.zeros((h, w), np.float32)

    if kind == "scratch":
        # Straight, no branching, no taper, often BRIGHT: the classic false positive.
        for _ in range(rng.integers(1, 4)):
            a = rng.uniform(0, np.pi)
            L = rng.uniform(0.4, 1.2) * max(h, w)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            p0 = (int(cx - L / 2 * np.cos(a)), int(cy - L / 2 * np.sin(a)))
            p1 = (int(cx + L / 2 * np.cos(a)), int(cy + L / 2 * np.sin(a)))
            cv2.line(layer, p0, p1, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
        bright = rng.random() < 0.55
    elif kind == "seam":
        # Weld seam / expansion joint: long, straight, wide, regular.
        a = rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.05)
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        L = 2 * max(h, w)
        p0 = (int(cx - L * np.cos(a)), int(cy - L * np.sin(a)))
        p1 = (int(cx + L * np.cos(a)), int(cy + L * np.sin(a)))
        cv2.line(layer, p0, p1, 1.0, int(rng.integers(3, 9)), cv2.LINE_AA)
        bright = False
    elif kind == "grain":
        # Wood grain / brushed metal: many near-parallel lines.
        a = rng.uniform(0, np.pi)
        for _ in range(rng.integers(6, 22)):
            off = rng.uniform(-1, 1) * max(h, w) * 0.5
            cx, cy = w / 2 - off * np.sin(a), h / 2 + off * np.cos(a)
            L = 2 * max(h, w)
            p0 = (int(cx - L * np.cos(a)), int(cy - L * np.sin(a)))
            p1 = (int(cx + L * np.cos(a)), int(cy + L * np.sin(a)))
            cv2.line(layer, p0, p1, rng.uniform(0.3, 1.0), 1, cv2.LINE_AA)
        bright = False
    elif kind == "grout":
        # Tile grout / brick mortar: an orthogonal grid of straight channels.
        step = int(rng.integers(40, 110))
        off = int(rng.integers(0, step))
        thick = int(rng.integers(2, 6))
        for x in range(off, w, step):
            cv2.line(layer, (x, 0), (x, h), 1.0, thick, cv2.LINE_AA)
        for y in range(off, h, step):
            cv2.line(layer, (0, y), (w, y), 1.0, thick, cv2.LINE_AA)
        bright = rng.random() < 0.4
    elif kind == "cable":
        # Wire or its shadow: a smooth catenary, constant width, no branching.
        pts = []
        x0, x1 = -20, w + 20
        y0 = rng.uniform(0.15, 0.85) * h
        sag = rng.uniform(-0.25, 0.25) * h
        for t in np.linspace(0, 1, 40):
            pts.append((x0 + t * (x1 - x0), y0 + sag * np.sin(np.pi * t)))
        cv2.polylines(layer, [np.asarray(pts, np.int32)], False, 1.0,
                      int(rng.integers(2, 6)), cv2.LINE_AA)
        bright = rng.random() < 0.3
    else:  # spatter -- dark blobs and streaks (dirt, oil, water stains)
        for _ in range(rng.integers(3, 14)):
            cv2.circle(layer, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                       int(rng.uniform(2, 14)), rng.uniform(0.3, 1.0), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.0)
        bright = False

    layer = cv2.GaussianBlur(layer, (0, 0), rng.uniform(0.5, 1.5))
    amp = rng.uniform(0.15, 0.55)
    if bright:
        out = img + (amp * 110.0) * layer[..., None]
    else:
        out = img * (1.0 - amp * layer[..., None])
    ns = rng.uniform(0.5, 3.0)
    out += rng.normal(0, ns, out.shape).astype(np.float32) * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), kind


# --------------------------------------------------------------- one sample
def make_one(job):
    # NOTE: out_dir travels in the job, never via a module global. On Windows the
    # pool uses spawn, so workers re-import this module fresh and would silently
    # fall back to the default path -- and cv2.imwrite reports a missing directory
    # by returning False, not by raising, so that failure is invisible.
    idx, run_seed, bg_path, bg_id, material, kind, out_dir = job
    out_dir = Path(out_dir)
    rng = np.random.default_rng([run_seed, idx])   # per-sample independent stream

    bg = cv2.imread(str(ROOT / bg_path), cv2.IMREAD_COLOR)
    if bg is None:
        return None
    if bg.shape[0] != RENDER_SIZE or bg.shape[1] != RENDER_SIZE:
        bg = cv2.resize(bg, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)

    # light geometric augmentation of the substrate (never of the label alone)
    if rng.random() < 0.5:
        bg = cv2.flip(bg, int(rng.integers(-1, 2)))
    k = int(rng.integers(0, 4))
    if k:
        bg = np.rot90(bg, k).copy()

    pid = f"{kind}_{idx:07d}"
    if kind == "negative":
        img, sub = draw_negative(bg, rng, material)
        mask = np.zeros(img.shape[:2], np.uint8)
        meta = dict(neg_kind=sub)
    else:
        grain = float(rng.uniform(0, np.pi)) if material == "wood" else None
        scale = float(rng.uniform(0.55, 2.0))       # camera-distance proxy
        polys = gen_crack((RENDER_SIZE, RENDER_SIZE), material, rng,
                          scale=scale, grain_angle=grain)
        rp = random_params(rng, material)
        img, mask = render(bg, polys, rp, rng)
        if (mask > 0).mean() < 0.0006:              # too faint to be a usable label
            return None
        meta = dict(scale=scale, grain=grain, **{f"rp_{k}": v
                                                 for k, v in asdict(rp).items()})

    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    ok_i = cv2.imwrite(str(out_dir / "images" / f"{pid}.png"), img)
    ok_m = cv2.imwrite(str(out_dir / "masks" / f"{pid}.png"), mask)
    if not (ok_i and ok_m):
        raise IOError(f"imwrite failed for {pid} under {out_dir} "
                      f"(images={ok_i} masks={ok_m})")

    return dict(patch_id=pid, kind=kind, material=material, bg_id=bg_id,
                bg_path=bg_path, rng_seed=run_seed, sample_index=idx,
                generator_version=GENERATOR_VERSION,
                crack_px=int((mask > 0).sum()),
                crack_px_frac=float((mask > 0).mean()),
                meta=json.dumps(meta, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-crack", type=int, default=100_000)
    ap.add_argument("--n-negative", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--materials", nargs="*", default=None,
                    help="regenerate only these materials (width fixes)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    bgs = pd.read_csv(BG_CSV)
    bgs = bgs[bgs.material != "unknown"]
    if args.materials:
        bgs = bgs[bgs.material.isin(args.materials)]
        print(f'[gen] restricted to {sorted(set(bgs.material))}')
    print(f"[gen] {len(bgs)} backgrounds across {bgs.material.nunique()} materials")

    rng = np.random.default_rng(args.seed)
    # Sample backgrounds per material with equal weight per MATERIAL, not per image,
    # so a material with 2,500 backgrounds does not drown one with 94.
    mats = sorted(bgs.material.unique())
    jobs = []
    for i in range(args.n_crack + args.n_negative):
        kind = "crack" if i < args.n_crack else "negative"
        m = mats[int(rng.integers(0, len(mats)))]
        pool = bgs[bgs.material == m]
        r = pool.iloc[int(rng.integers(0, len(pool)))]
        jobs.append((i, args.seed, r["path"], r["bg_id"], m, kind, str(out_dir)))

    print(f"[gen] {args.n_crack} crack + {args.n_negative} negative "
          f"-> {out_dir}  ({args.workers} workers)")
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(make_one, jobs, chunksize=64):
            done += 1
            if r:
                rows.append(r)
            if done % 5000 == 0:
                print(f"  ... {done}/{len(jobs)} kept={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    prov = out_dir.parent / f"{out_dir.name}_provenance.parquet"
    df.to_parquet(prov, index=False)
    print(f"\n[gen] kept {len(df)} / {len(jobs)} "
          f"({100*len(df)/max(len(jobs),1):.1f}%)")
    print(df.groupby(["material", "kind"]).size().to_string())
    print(f"[gen] mean fg on cracks: "
          f"{100*df[df.kind=='crack'].crack_px_frac.mean():.2f}%")

if __name__ == "__main__":
    main()
