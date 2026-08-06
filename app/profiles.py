"""The post-processing profile — the one place threshold values exist.

Split out of `postprocess.py` for one reason: the interface has to *display* the active
thresholds (Phase 2 figure 8) and a test has to prove it did not copy them
(Phase 2 T-06), but `postprocess` imports cv2, numpy and scikit-image, which the
mock-mode deployment deliberately does not install. A UI that could not import the
module would have had to hold its own copy of the numbers, and the first re-tune would
have made the screen lie.

So this module is stdlib-only and imports nothing. `postprocess` re-exports `Profile`
and `ACTIVE_PROFILE` from here, so there is still exactly one definition and
`postprocess.Profile is profiles.Profile` holds — `tests/test_materials.py` asserts it.

Re-tuning a threshold means editing this file and nothing else. The database records
the values that produced each stored inspection, so old results keep resolving to the
thresholds they were actually measured under.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    """Versioned post-processing config, recorded with every result.

    crack_thresh / scratch_thresh
        Per-class probability floors. The class decision is **not** argmax: a pixel
        below both floors is background, and where both pass, the higher score wins
        (ARCHITECTURE §7.1). Chosen on the validation split by
        `bench/class_thresh.py`, never on a test split.

    min_area_px / min_skeleton_px
        Region floors, derived from the clean-surface false-positive budget (NFR-03),
        not from tuning against test images.

    version_no
        Bumped whenever any value above changes, so a stored inspection still resolves
        to the thresholds that produced it.

    v3 (crack 0.40 -> 0.30).  v2 was selected by a sweep that never saw a scratch and
    covered only the first 43 % of `val`: evaluation splits are walked in manifest
    order, so `--eval-batches` takes a prefix rather than a sample, and every scratch
    row sits past the cut. Re-run over the whole split, the same selection rule picks
    0.30. On `test_factory` that moves crack recall 0.660 -> 0.733 for -0.003 clDice,
    and false-positive area on all 4,626 negatives goes 0.00162 -> 0.00191, well inside
    the 0.005 NFR-03 budget. The scratch floor stays 0.20 -- unchanged in value, but
    now chosen on evidence instead of being the first grid point among ties.
    """

    profile_id: str = "conveyor-v3"
    crack_thresh: float = 0.30
    scratch_thresh: float = 0.20
    min_area_px: int = 24
    min_skeleton_px: int = 6
    connectivity: int = 8
    version_no: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


#: The active profile, read by the application at start-up and by the seed.
ACTIVE_PROFILE = Profile()
