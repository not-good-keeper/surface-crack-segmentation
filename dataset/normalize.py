"""Filter the indexed corpus to what is safe to train on.

Drops skeleton-annotated sources (near-zero width variance means the mask is a dilated
centreline, not the crack) and near-duplicates, which leak across splits because the
group check sees different parent ids for the same picture.

Writes data/manifest_clean.csv and a dropped-rows report; nothing leaves disk.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data/manifest.csv"
OUT = ROOT / "data/manifest_clean.csv"
REPORT = ROOT / "data/report/normalize.txt"

SUB_PREFIXES = ["noncrack", "CRACK500", "GAPS", "cracktree", "CFD", "DeepCrack",
                "Rissbilder", "Volker", "Ceramic"]


def subsource(name: str) -> str:
    src = name.split("__", 1)[0]
    s = name.split("__", 1)[1] if "__" in name else name
    # Scratch rows must be separable from their source's cracks by subsource alone:
    # the split stage quarantines MVTec's scratch masks (median width 33.6 px, an
    # anomaly-region outline rather than the scratch itself) into a detection-only
    # split, and it can only find them if they carry a distinct subsource here.
    if s.startswith("scratch_"):
        return f"{src}_scratch"
    for p in SUB_PREFIXES:
        if s.startswith(p):
            return p
    if s[:2] in ("a_", "b_", "c_", "d_"):
        return "masonry_abcd"
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-var-thresh", type=float, default=0.05,
                    help="std/mean of crack width below this => constant-width mask")
    ap.add_argument("--min-median-width", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(IN)
    df["subsource"] = df.name.map(subsource)
    n0 = len(df)
    lines = [f"input rows: {n0}"]

    # ---- 1. skeleton detection, by measurement -----------------------------
    pos = df[(df.role == "positive") & (df.crack_px > 0)]
    drop_subs = []
    lines.append("\n--- width audit (positives) ---")
    lines.append(f"{'subsource':<16}{'n':>6}{'median_w':>10}{'std/mean':>10}  verdict")
    for sub, g in pos.groupby("subsource"):
        med = g.mean_crack_width.median()
        cv = g.mean_crack_width.std() / max(g.mean_crack_width.mean(), 1e-6)
        skeleton = (cv < args.width_var_thresh) or (med < args.min_median_width and cv < 0.25)
        if skeleton:
            drop_subs.append(sub)
        lines.append(f"{sub:<16}{len(g):>6}{med:>10.2f}{cv:>10.4f}  "
                     f"{'SKELETON -> drop' if skeleton else 'keep'}")

    before = len(df)
    df = df[~df.subsource.isin(drop_subs)]
    lines.append(f"\ndropped {before-len(df)} rows from skeleton sources: {drop_subs}")

    # ---- 2. near-duplicates -------------------------------------------------
    before = len(df)
    has = df.phash.notna() & (df.phash.astype(str) != "")
    dup_mask = has & df.duplicated("phash", keep="first")
    dropped_dupes = int(dup_mask.sum())
    df = df[~dup_mask]
    lines.append(f"dropped {dropped_dupes} near-duplicate rows (identical phash)")

    # ---- 3. material refinement --------------------------------------------
    # Sources label material coarsely. Refining here keeps it reproducible; these
    # corrections used to be hand edits to the split manifest.
    before_mat = df.material.copy()
    # Rendered plaster walls, not poured concrete -- 3.1k rows, enough to skew every
    # per-material number.
    df.loc[df.subsource.isin(["Rissbilder", "Volker"]), "material"] = "plaster"
    # Epoxy resin embedding a commutator, not a plastic housing.
    df.loc[df.source == "kolektor1", "material"] = "epoxy"
    # A pharmaceutical pill, not a plastic moulding. Its own material keeps the data
    # while excluding it from FACTORY_MATERIALS.
    is_capsule = df.name.str.startswith("mvtec__capsule")
    df.loc[is_capsule, "material"] = "pharma_capsule"
    n_relabelled = int((before_mat != df.material).sum())
    lines.append(f"\nrelabelled material on {n_relabelled} rows "
                 f"(Rissbilder/Volker -> plaster, kolektor1 -> epoxy, "
                 f"mvtec capsule -> pharma_capsule [{int(is_capsule.sum())} rows])")

    # ---- summary ------------------------------------------------------------
    lines.append(f"\nkept {len(df)} / {n0} rows")
    lines.append("\n--- surviving positives by material ---")
    p = df[(df.role == "positive") & (df.crack_px > 0)]
    for m, g in p.groupby("material"):
        lines.append(f"  {m:<10} {len(g):>6}")
    lines.append("\n--- roles ---")
    for r, g in df.groupby("role"):
        lines.append(f"  {r:<15} {len(g):>6}")

    df.to_csv(OUT, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
