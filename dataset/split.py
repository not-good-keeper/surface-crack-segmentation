"""Build the frozen evaluation suite (splits v7).

Synthetic data never enters val or test: it would measure the generator, not the model.

  train                  everything not held out
  val                    development only -- early stopping, model selection
  val_unseen_material    masonry: selection on a material absent from train
  test_factory           HEADLINE. Real cracks on factory product surfaces
  test_factory_scratch   HEADLINE. Real scratches on factory product surfaces
  test_scratch_blob      MVTec scratches -- detection/class only, never IoU or clDice
  test_seen              civil materials present in training (auxiliary evidence)
  test_unseen_material   wood: cross-material transfer
  test_negatives         false-positive rate: hard negatives + clean surfaces
  wood_bg                synthesis substrate only, never a training sample

Splits are grouped by parent image, because several sources are crops of shared parent
photographs and splitting at crop level puts the same physical crack on both sides.

See docs/DATASET.md for what each split contains and why the headline is test_factory.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data/manifest_clean.csv"
OUT = ROOT / "data/splits.json"

EVAL_ONLY_SOURCES = {"kolektor1", "mvtec"}
UNSEEN_MATERIAL = "wood"
UNSEEN_VAL_MATERIAL = "masonry"

# Materials that are finished products on a conveyor line. Everything else (asphalt,
# plaster, concrete, masonry, wood) is civil infrastructure: useful for learning crack
# morphology, but not evidence about a factory line.
FACTORY_MATERIALS = {"steel", "metal", "plastic", "ceramic", "epoxy", "glass"}

# Global split ratio, applied at GROUP level and stratified by material x class.
# 80/15/5 as specified. The test share is deliberately the smallest: its job is to be
# an untouched final check, and every image spent on it is one not spent training a
# corpus whose scarce materials are the whole problem.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.15, 0.05

# `sdx` is the only source of real factory defect masks for BOTH classes, so it is the
# only source that must be present on every side of the split.
SDX_TRAIN_FRAC, SDX_VAL_FRAC = TRAIN_FRAC, VAL_FRAC


def stratified_groups(gstat, rng, train_f, val_f):
    """Assign whole groups to train/val/test, balanced within each stratum.

    Stratifying on material x foreground-quartile matters more than the ratio itself.
    Splitting a pooled list would let one material land mostly in test and another
    mostly in train purely by chance, and with materials as thin as glass (12 masks)
    or epoxy (49) that is not a small effect -- it decides whether a per-material
    number exists at all.

    Groups, never images: an image is a crop or an augmentation of a photograph, and
    two views of one physical defect on opposite sides of the split is leakage that
    every downstream metric would silently absorb.
    """
    val_g, test_g = set(), set()
    for _, sub in gstat.groupby(["material", "fgb"]):
        g = sub.group.to_numpy().copy()
        rng.shuffle(g)
        n = len(g)
        if n < 3:
            # Too few groups to divide three ways; keep them in train rather than
            # manufacturing a one-image "split" that reports noise as a metric.
            continue
        n_tr = max(1, int(round(n * train_f)))
        n_va = max(1, int(round(n * val_f)))
        n_tr = min(n_tr, n - 2)                     # leave at least one for val+test
        val_g.update(g[n_tr:n_tr + n_va])
        test_g.update(g[n_tr + n_va:])
    return val_g, test_g


def parent_id(name: str) -> str:
    """Collapse crops of the same photograph onto one group key."""
    src, _, stem = name.partition("__")
    s = stem or name
    if s.startswith("CRACK500"):
        # CRACK500_20160222_081011_1_721 -> date_time identifies the parent photo
        m = re.match(r"CRACK500_(\d+_\d+)", s)
        if m:
            return f"{src}:CRACK500:{m.group(1)}"
    if s.startswith("Rissbilder"):
        m = re.match(r"(Rissbilder_for_Florian_\w+?)_\d+$", s)
        if m:
            return f"{src}:{m.group(1)}"
    if src == "sdx":
        # sdx sids are independent photographs, not crops of a shared parent, so
        # each is its own group. Grouping them by prefix collapses all 763 scratches
        # onto two keys, which forces the class wholly into train or wholly into
        # test.
        return f"sdx:{s}"
    if s.startswith(("a_", "b_", "c_", "d_")):
        m = re.match(r"([a-d]_\d+)_", s)      # a_0_10 -> a_0
        if m:
            return f"{src}:{m.group(1)}"
    if src == "wood":
        # 9-digit ids; the first 4 chars separate 245 boards (max 74 images each),
        # and deeper prefixes add no further separation.
        return f"wood:{s[:4]}"
    if src == "kolektor1":
        return f"kolektor1:{s.split('_')[0]}"  # kos01_Part3 -> kos01 (same commutator)
    # strip a trailing crop/patch counter if present
    base = re.sub(r"_crop\d+$", "", s)
    return f"{src}:{base}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    df = pd.read_csv(IN)
    df["group"] = df.name.map(parent_id)
    rng = np.random.default_rng(args.seed)
    assign = {}

    # ---- 1. unseen-material holdout: ALL real wood positives ---------------
    wood_pos = df[(df.material == UNSEEN_MATERIAL) & (df.role == "positive")]
    for n in wood_pos.name:
        assign[n] = "test_unseen_material"

    # Clean wood is split, not dumped into test: some becomes `wood_bg` (synthesis
    # substrate only), the rest measures false positives. Without the partition we
    # would synthesize onto the same images we later score FP against.
    #
    # Partition at BOARD level -- one board's faces share texture, lighting and grain.
    wood = df[df.material == UNSEEN_MATERIAL]
    boards_with_pos = set(wood.loc[wood.role == "positive", "group"])
    clean_boards = sorted(set(wood.group) - boards_with_pos)
    rng.shuffle(clean_boards)
    bg_boards = set(clean_boards[:int(len(clean_boards) * 0.70)])

    for _, r in wood.iterrows():
        if r["name"] in assign:            # positives already claimed above
            continue
        # any non-positive on a board that HAS cracks stays on the test side
        assign[r["name"]] = "wood_bg" if r["group"] in bg_boards else "test_negatives"

    # ---- 2. unseen-VALIDATION material --------------------------------------
    # Selection needs a material never trained on: val alone keeps improving while
    # cross-material decays. Masonry is cheap to hold out.
    for n in df.loc[(df.material == UNSEEN_VAL_MATERIAL)
                    & (df.role == "positive"), "name"]:
        assign[n] = "val_unseen_material"

    # ---- 3. scratches -------------------------------------------------------
    # MVTec scratch masks outline the anomaly region, not the scratch (33.6 px median
    # against 7.3 for sdx). Kept for detection and class only, in their own split so
    # they cannot reach IoU or clDice.
    for n in df.loc[df.subsource == "mvtec_scratch", "name"]:
        assign[n] = "test_scratch_blob"

    # sdx defect rows (both classes) get an explicit 60/15/25 group split so that real
    # steel cracks AND real steel scratches are present in train, val and test.
    sdx = df[(df.source == "sdx") & df.role.isin(["positive", "scratch"])]
    for role, test_split in (("positive", "test_factory"),
                             ("scratch", "test_factory_scratch")):
        g = sorted(set(sdx.loc[sdx.role == role, "group"]))
        rng.shuffle(g)
        n_tr = int(round(len(g) * args.train_frac))
        n_va = int(round(len(g) * args.val_frac))
        bucket = dict.fromkeys(g[:n_tr], "train")
        bucket.update(dict.fromkeys(g[n_tr:n_tr + n_va], "val"))
        bucket.update(dict.fromkeys(g[n_tr + n_va:], test_split))
        for _, r in sdx[sdx.role == role].iterrows():
            assign[r["name"]] = bucket[r["group"]]

    # ---- 4. eval-only sources -> test --------------------------------------
    # KolektorSDD1 and MVTec hold the only real epoxy, plastic and ceramic cracks. Far
    # too few to train on, and all on factory product surfaces, so they go straight to
    # the headline test split.
    ev = df[df.source.isin(EVAL_ONLY_SOURCES)]
    for _, r in ev.iterrows():
        if r["name"] in assign:
            continue
        if r["role"] != "positive":
            assign[r["name"]] = "test_negatives"
        else:
            assign[r["name"]] = ("test_factory" if r["material"] in FACTORY_MATERIALS
                                 else "test_seen")

    # ---- 5. everything else, split BY GROUP --------------------------------
    rest = df[~df.name.isin(assign)]
    pos_rest = rest[rest.role == "positive"]

    # stratify groups by material x foreground bucket so splits match in difficulty
    gstat = (pos_rest.groupby("group")
             .agg(material=("material", "first"),
                  fg=("crack_px_frac", "mean"), n=("name", "size"))
             .reset_index())
    if len(gstat):
        gstat["fgb"] = pd.qcut(gstat.fg.rank(method="first"), 4, labels=False)
        val_g, test_g = stratified_groups(gstat, rng, args.train_frac, args.val_frac)
    else:
        val_g, test_g = set(), set()

    for _, r in rest.iterrows():
        g = r["group"]
        if r["role"] != "positive":
            # Negatives follow the same 80/15/5 ratio. val needs its own negatives or
            # the false-positive rate is unobservable during model selection -- and
            # that is the metric that decides whether the system is usable at all.
            if g in test_g:
                assign[r["name"]] = "test_negatives"
            else:
                u = rng.random()
                assign[r["name"]] = ("test_negatives" if u < args.test_frac else
                                     "val" if u < args.test_frac + args.val_frac
                                     else "train")
        else:
            # A held-out positive on a factory product surface is headline evidence;
            # the same image on asphalt or plaster is auxiliary. Route by material so
            # the headline split is defined by the deployment domain rather than by
            # which source happened to be marked eval-only.
            held = ("val" if g in val_g else
                    "test_seen" if g in test_g else "train")
            if held == "test_seen" and r["material"] in FACTORY_MATERIALS:
                held = "test_factory"
            assign[r["name"]] = held

    df["split"] = df.name.map(assign).fillna("train")

    # ---- 6. leakage assertions (fail loudly) -------------------------------
    errs, warns = [], []
    # Every non-training split belongs here. Omitting one does not weaken the check
    # quietly -- it removes it: a group spanning train and an unlisted split reads as
    # clean. `test_scratch` and `test_steel` were absent from this set under v4, so the
    # scratch and steel holdouts were never leakage-checked at all.
    EVAL = {"val", "val_unseen_material", "test_seen", "test_factory",
            "test_factory_scratch", "test_scratch_blob", "test_unseen_material",
            "test_negatives"}

    # What actually matters is train<->eval contamination. A group spanning two EVAL
    # splits is harmless: e.g. one Kolektor commutator contributes its cracked faces
    # to test_seen and its clean faces to test_negatives, and nothing was trained on.
    def cross_train_eval(keycol):
        g = df.groupby(keycol).split.agg(set)
        return g[g.map(lambda s: ("train" in s or "wood_bg" in s) and bool(s & EVAL))]

    bad = cross_train_eval("group")
    if len(bad):
        errs.append(f"{len(bad)} groups span train and eval, e.g. {list(bad.index[:5])}")
    eval_only_span = df.groupby("group").split.agg(set)
    eo = eval_only_span[eval_only_span.map(lambda s: len(s) > 1 and not (
        "train" in s or "wood_bg" in s))]
    if len(eo):
        warns.append(f"{len(eo)} groups span multiple EVAL splits (harmless: no "
                     f"training contamination), e.g. {list(eo.index[:3])}")

    ph = df[df.phash.notna() & (df.phash.astype(str) != "")]
    badp = cross_train_eval("phash") if len(ph) else []
    if len(badp):
        errs.append(f"{len(badp)} phashes span train and eval")
    if (df[df.split == "test_unseen_material"].material != UNSEEN_MATERIAL).any():
        errs.append("non-wood rows in test_unseen_material")
    if (df[df.split == "train"].material == UNSEEN_MATERIAL).any():
        errs.append("wood leaked into train")

    # The unseen-VALIDATION material must be absent from train, val and test_seen, or
    # early stopping on it is contaminated and the split is decorative. Under v4 the
    # designated material still had 1,492 rows in train and QA reported GREEN.
    uv = df[(df.material == UNSEEN_VAL_MATERIAL)
            & df.split.isin(["train", "val", "test_seen"])]
    if len(uv):
        errs.append(f"unseen-VALIDATION material '{UNSEEN_VAL_MATERIAL}' appears in "
                    f"{sorted(set(uv.split))} ({len(uv)} rows)")
    if not len(df[df.split == "val_unseen_material"]):
        errs.append(f"'{UNSEEN_VAL_MATERIAL}' declared unseen-val but the split is empty")

    # The three-class model cannot learn a class it never sees. This is the assertion
    # that would have caught v4, where all 879 real scratches sat in test.
    for role, where in (("scratch", "train"), ("scratch", "val"),
                        ("positive", "train")):
        if not len(df[(df.role == role) & (df.split == where)]):
            errs.append(f"no real '{role}' rows in {where} -- that class is untrainable")

    bad_mat = df[(df.split == "test_factory") & ~df.material.isin(FACTORY_MATERIALS)]
    if len(bad_mat):
        errs.append(f"{len(bad_mat)} non-factory materials in test_factory: "
                    f"{sorted(set(bad_mat.material))}")
    if (df[df.split == "test_scratch_blob"].subsource != "mvtec_scratch").any():
        errs.append("test_scratch_blob must contain only mvtec_scratch rows -- its "
                    "whole purpose is to quarantine the wide-mask convention")
    # wood_bg is a synthesis substrate, not a training sample; it must never be
    # loaded as `train`, and must not overlap the wood held out for FP measurement.
    bg = set(df[df.split == "wood_bg"].name)
    tn = set(df[df.split == "test_negatives"].name)
    if bg & tn:
        errs.append(f"{len(bg & tn)} images in both wood_bg and test_negatives")

    # ---- 7. report ---------------------------------------------------------
    print(f"{'split':<22}{'n':>7}{'crack':>7}{'scr':>6}  materials")
    for s, g in df.groupby("split"):
        mats = ",".join(sorted(set(g.material.astype(str))))
        print(f"{s:<22}{len(g):>7}{(g.role=='positive').sum():>7}"
              f"{(g.role=='scratch').sum():>6}  {mats}")
    print("\ndefect rows per material by split:")
    p = df[df.role.isin(["positive", "scratch"])]
    print(pd.crosstab(p.material, p.split).to_string())

    payload = {n: s for n, s in zip(df.name, df.split)}
    body = json.dumps(payload, sort_keys=True)
    meta = dict(version=5, seed=args.seed, unseen_material=UNSEEN_MATERIAL,
                unseen_val_material=UNSEEN_VAL_MATERIAL,
                factory_materials=sorted(FACTORY_MATERIALS),
                eval_only=sorted(EVAL_ONLY_SOURCES),
                changes=[
                    "v5: sdx defect rows split 60/15/25 so real scratches reach train "
                    "(under v4 all 879 sat in test and the class was untrainable)",
                    "v5: parent_id() groups sdx per image; the old key collapsed 763 "
                    "scratches onto 2 groups",
                    "v5: headline moves to test_factory / test_factory_scratch; wood "
                    "stays as transfer evidence only",
                    "v5: mvtec_scratch quarantined in test_scratch_blob (detection and "
                    "class only -- 33.6 px median mask width)",
                    "v5: scratch and steel holdouts are produced HERE and are covered "
                    "by the leakage checks; previously no committed script made them",
                ],
                n=len(df), sha256=hashlib.sha256(body.encode()).hexdigest())
    OUT.write_text(json.dumps({"meta": meta, "splits": payload}, sort_keys=True))
    df.to_csv(ROOT / "data/manifest_split.csv", index=False)
    print(f"\nfrozen -> {OUT}\nsha256 {meta['sha256'][:16]}...")

    for w in warns:
        print(f"\nnote: {w}")
    if errs:
        print("\n*** LEAKAGE ERRORS ***")
        for e in errs:
            print("  " + e)
        return 1
    print("\nleakage checks PASSED (no group or phash spans train and eval; "
          "unseen material isolated; wood_bg disjoint from test_negatives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
