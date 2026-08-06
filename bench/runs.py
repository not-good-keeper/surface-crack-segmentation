"""Load a stored run and its weights.

Matched on the `tag` field inside each JSON, never by globbing filenames: a glob over
`*V3_*` once pulled in thirty unrelated runs. The filename is a label, the tag is the key.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/bench"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402


def find(tag):
    """-> the run dict for `tag`.

    A match needs `args` as well as the tag. Evaluation reports live in the same
    directory and carry the tag of the run they describe, so matching on tag alone
    picked up FULLEVAL_<tag>.json -- which sorts first and has no `args` -- and every
    caller died on a KeyError far from the cause.
    """
    for p in sorted(BENCH.glob("*.json")):
        try:
            run = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(run, dict) and run.get("tag") == tag and "args" in run:
            return run
    raise SystemExit(f"no run tagged {tag!r} in {BENCH} (a run record needs both "
                     f"'tag' and 'args'; evaluation reports have only 'tag')")


def load_weights(path, model_name, classes, device=None, **ds_kw):
    """-> (model, {}, ds_kw) for a checkpoint with no run JSON beside it yet.

    `.best.pt` is written every time a run improves, so this is how a run still in
    flight can be evaluated. It wraps the state dict in a dict; a final `.pt` does not.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    model = M.build(model_name, classes=classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, {}, dict(classes=classes, **ds_kw)


def load(tag, device=None):
    """-> (model in eval mode, run dict, dataset kwargs).

    Dataset kwargs come from the run's own arguments, not from defaults: an evaluation
    must not silently score a checkpoint under a profile or transform it never saw.
    """
    run = find(tag)
    ck = run.get("checkpoint")
    if not ck or not Path(ck).exists():
        raise SystemExit(f"run {tag} has no saved checkpoint (train with --save)")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = int(run["args"].get("classes", 1))
    model = M.build(run["model"], classes=classes).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    model.eval()
    ds_kw = dict(classes=classes,
                 camera_profile=run["args"].get("camera_profile", "conveyor"),
                 prep=run["args"].get("prep"),
                 resize=bool(run["args"].get("resize", False)))
    return model, run, ds_kw
