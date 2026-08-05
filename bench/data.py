"""Patch dataset built from the frozen splits.

Rationale for the sampling choices lives in docs/DECISIONS.md (ADR-012 to ADR-014).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"

BACKGROUND, CRACK, SCRATCH = 0, 1, 2
# Class comes from the manifest, not the mask: masks are binary on disk, so defect type
# is a property of the row.
ROLE_CLASS = {"positive": CRACK, "scratch": SCRATCH}
KIND_CLASS = {"crack": CRACK, "scratch": SCRATCH}


class CrackPatches(Dataset):
    def __init__(self, split, train=False, classes=1, size=256, length=None,
                 seed=0, pos_frac=0.65, exclude_materials=(),
                 camera_aug=False, camera_profile="conveyor",
                 synth_dir=None, synth_frac=0.5, synth_kinds=None,
                 scratch_frac=0.35, prep=None, resize=False):
        self.size, self.train, self.pos_frac = size, train, pos_frac
        # Whole frame scaled to `size`, instead of a crack-centred crop at native scale.
        # The crop path trains and scores on windows that always contain a defect, so it
        # never sees the rest of the part -- applied to a whole surface the model painted
        # 14x the true defect area. Resize is what app/inference.py does, so this makes
        # the measured path and the deployed path the same one (T-02).
        self.resize = resize
        # Built lazily: bench/preprocess.py composes ops into a closure, and Windows
        # spawns dataloader workers by pickling this object.
        self.prep, self._prep_fn = prep, None
        self.camera_aug = camera_aug and train      # never augment evaluation data
        self.camera_profile = camera_profile
        self.classes = classes
        self.rng = np.random.default_rng(seed)

        df = pd.read_csv(ROOT / "data/manifest_split.csv")
        df = df[df.split == split]
        if exclude_materials:
            df = df[~df.material.isin(exclude_materials)]
        # Under the binary head a scratch is genuinely not-a-crack, so it belongs in the
        # negative pool. Only the three-class head can carry it as a positive.
        defect_roles = (["positive", "scratch"] if classes > 1 else ["positive"])
        self.pos = df[df.role.isin(defect_roles)
                      & (df.crack_px > 0)].reset_index(drop=True)
        self.neg = df[~df.role.isin(defect_roles)].reset_index(drop=True)

        # Scratch is ~6 % of positives, too rare to learn at its natural rate. Quota it
        # at the sampler, not in the loss (ADR-012). Index arrays rather than copies:
        # each worker forks this object, and duplicating the frames exhausts the
        # Windows commit limit.
        self.scratch_frac = scratch_frac
        self.pos_by_cls = {}
        if classes > 1:
            role = self.pos.role.to_numpy()
            self.pos_by_cls = {CRACK: np.flatnonzero(role == "positive"),
                               SCRATCH: np.flatnonzero(role == "scratch")}

        self.synth = None
        self.synth_by_kind = {}
        if synth_dir:
            if not train:
                raise ValueError("synthetic data must never enter val/test")
            self._load_synth(synth_dir, synth_kinds, synth_frac,
                             exclude_materials, classes)

        self.length = length or (len(self.pos) * 3 if train else self._eval_len())

    def _load_synth(self, synth_dir, synth_kinds, synth_frac,
                    exclude_materials, classes):
        """Comma-separated dirs, each optionally `dir:kind|kind` to pick kinds per
        source: cracks from our compositor, scratches from DefectForge (ADR-013)."""
        g_want = ({k.strip() for k in synth_kinds.split(",") if k.strip()}
                  if synth_kinds else None)
        parts = []
        for d in str(synth_dir).split(","):
            d = d.strip()
            if not d:
                continue
            want = g_want
            if ":" in d:
                d, spec = d.split(":", 1)
                want = {k.strip() for k in spec.split("|") if k.strip()}
            sp = ROOT / d
            prov = sp.parent / f"{sp.name}_provenance.parquet"
            if not prov.exists():
                print(f"[data] WARNING: no provenance for {d}, skipping")
                continue
            q = pd.read_parquet(prov)
            before = len(q)
            if want:
                q = q[q.kind.isin(want)]
            q["_src_dir"] = str(sp)
            print(f"[data] {d}: kept {len(q)}/{before} "
                  f"kinds={sorted(want) if want else 'all'}")
            parts.append(q)
        if not parts:
            raise FileNotFoundError(f"no usable synthetic source in {synth_dir!r}")

        s = pd.concat(parts, ignore_index=True)
        if exclude_materials:
            s = s[~s.material.isin(exclude_materials)]

        # A patch inherits its background's split. Re-derive from the current manifest
        # rather than trusting backgrounds.csv, which predates the last re-cut.
        msdf = pd.read_csv(ROOT / "data/manifest_split.csv")[["name", "split"]]
        held = set(msdf.loc[~msdf.split.isin(["train", "wood_bg"]), "name"])
        if held and "bg_id" in s.columns:
            base = s.bg_id.astype(str).str.replace(r"_(crop|c)\d+$", "", regex=True)
            bad = base.isin(held)
            if bad.any():
                print(f"[data] dropped {int(bad.sum())} synthetic patches whose "
                      f"background is in a held-out split")
            s = s[~bad]

        self.synth = s.reset_index(drop=True)
        self.synth_dir = sp
        self.synth_frac = synth_frac
        self.synth_by_kind = ({str(k): np.flatnonzero(self.synth.kind.to_numpy() == k)
                               for k in self.synth.kind.unique()}
                              if classes > 1 else {})

    def _eval_len(self):
        # test_negatives has no positives; the interleaved formula would return 0 and
        # silently evaluate nothing.
        if not len(self.pos):
            return len(self.neg)
        if not len(self.neg):
            return len(self.pos)
        return len(self.pos) + min(len(self.neg), len(self.pos))

    def __len__(self):
        return self.length

    def _load(self, name):
        img = cv2.imread(str(CLEAN / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(CLEAN / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if msk is None and img is not None:
            msk = np.zeros(img.shape[:2], np.uint8)
        return img, msk

    def _crop(self, img, msk, rng, centre_on_crack):
        h, w = msk.shape
        s = self.size
        if h < s or w < s:
            pad = ((0, max(0, s - h)), (0, max(0, s - w)))
            img = np.pad(img, pad + ((0, 0),), mode="reflect")
            msk = np.pad(msk, pad, mode="constant")
            h, w = msk.shape
        if centre_on_crack and msk.any():
            # Uniform crops of a 400x400 image with 2-4 % foreground are mostly empty.
            ys, xs = np.nonzero(msk)
            i = int(rng.integers(len(ys)))
            cy = int(np.clip(ys[i] + rng.integers(-s // 3, s // 3), s // 2, h - s // 2))
            cx = int(np.clip(xs[i] + rng.integers(-s // 3, s // 3), s // 2, w - s // 2))
            y0, x0 = cy - s // 2, cx - s // 2
        else:
            y0 = int(rng.integers(0, h - s + 1))
            x0 = int(rng.integers(0, w - s + 1))
        return img[y0:y0 + s, x0:x0 + s], msk[y0:y0 + s, x0:x0 + s]

    def _aug(self, img, msk, rng):
        if rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if rng.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = int(rng.integers(0, 4))
        if k:
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        if rng.random() < 0.5:                       # photometric: image only
            img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.3)
                          + rng.uniform(-22, 22), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def _synth_item(self, rng):
        idx = None
        if self.synth_by_kind and rng.random() < self.scratch_frac:
            idx = self.synth_by_kind.get("scratch")
        k = (int(idx[rng.integers(len(idx))]) if idx is not None and len(idx)
             else int(rng.integers(len(self.synth))))
        r = self.synth.iloc[k]
        sdir = Path(r["_src_dir"]) if "_src_dir" in r else self.synth_dir
        img = cv2.imread(str(sdir / "images" / f"{r.patch_id}.png"))
        msk = cv2.imread(str(sdir / "masks" / f"{r.patch_id}.png"), 0)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
            msk = np.zeros((self.size, self.size), np.uint8)
        return img, msk, KIND_CLASS.get(str(r.kind), BACKGROUND)

    def _pick_row(self, i, rng):
        """-> (row, want_pos). Training samples; evaluation walks a fixed order."""
        if self.train:
            want_pos = rng.random() < self.pos_frac
            pool = self.pos if (want_pos and len(self.pos)) else self.neg
            sel = None
            if want_pos and self.pos_by_cls:
                want = SCRATCH if rng.random() < self.scratch_frac else CRACK
                cand = self.pos_by_cls.get(want)
                if cand is not None and len(cand):
                    sel = cand
            if not len(pool):
                pool = self.pos if len(self.pos) else self.neg
            j = (int(sel[rng.integers(len(sel))]) if sel is not None
                 else int(rng.integers(len(pool))))
        elif len(self.pos) and len(self.neg):
            want_pos = i % 2 == 0
            pool = self.pos if want_pos else self.neg
            j = (i // 2) % len(pool)
        else:
            # Single-class splits must not index (i//2) or half the images never run.
            want_pos = bool(len(self.pos))
            pool = self.pos if len(self.pos) else self.neg
            j = i % len(pool)
        return pool.iloc[j], want_pos

    def __getitem__(self, i):
        # Evaluation crops are deterministic so every model sees identical data.
        rng = self.rng if self.train else \
            np.random.default_rng((hash((i, self.train)) ^ 0x9E3779B9) & 0xFFFFFFFF)

        if self.train and self.synth is not None and rng.random() < self.synth_frac:
            img, msk, cls = self._synth_item(rng)
            img, msk = self._aug(img, msk, rng)
        else:
            row, want_pos = self._pick_row(i, rng)
            cls = ROLE_CLASS.get(str(row["role"]), BACKGROUND)
            img, msk = self._load(row["name"])
            if img is None:
                img = np.zeros((self.size, self.size, 3), np.uint8)
                msk = np.zeros((self.size, self.size), np.uint8)
            if self.resize:
                img = cv2.resize(img, (self.size, self.size), cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (self.size, self.size), cv2.INTER_NEAREST)
            else:
                img, msk = self._crop(img, msk, rng, centre_on_crack=want_pos)
            if self.train:
                img, msk = self._aug(img, msk, rng)

        if self.camera_aug:
            from camera_aug import apply as _cam
            img, msk = _cam(np.ascontiguousarray(img), np.ascontiguousarray(msk), rng,
                            profile=self.camera_profile)

        # After camera_aug, never before: the transform is what the app runs on a frame
        # the camera has already degraded, so it has to see the degradation.
        if self.prep:
            if self._prep_fn is None:
                from preprocess import build
                self._prep_fn = build(self.prep)
            img = self._prep_fn(np.ascontiguousarray(img))

        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.)
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        fg = msk > 127
        if self.classes == 1:
            return x, torch.from_numpy(fg.astype(np.float32)).unsqueeze(0)
        # Index map for cross-entropy, not stacked binary planes: softmax means a pixel
        # is background, crack or scratch, never two at once.
        y = np.zeros(fg.shape, np.int64)
        y[fg] = cls
        return x, torch.from_numpy(y)
