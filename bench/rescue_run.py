"""Make a killed run evaluable.

    python bench/rescue_run.py --tag V12_22 --prep bilateral --resize

`bench/train.py` writes its run JSON only when the loop finishes, and every evaluation
script resolves a checkpoint through `runs.find()`, which matches on that JSON. So a run
stopped by the clock leaves a perfectly good `.best.pt` that nothing can load -- which is
exactly the situation a deadline produces.

This writes the minimum JSON `runs.load()` reads (tag, model, args, checkpoint) and
unwraps `.best.pt` -- saved as {"state": ...} on every improvement -- into the bare state
dict a final `.pt` contains, so the loader needs no special case.

The result carries no history and no test-split section: it is a handle for evaluating
weights, not a record of a run. `bench/final_eval.py` produces the numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="smpslim_timm-mobilenetv3_small_100")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--prep", default=None)
    ap.add_argument("--resize", action="store_true")
    ap.add_argument("--camera-profile", default="conveyor")
    args = ap.parse_args()

    stem = f"{args.model}_{args.tag}"
    best = BENCH / f"{stem}.best.pt"
    final = BENCH / f"{stem}.pt"
    if not best.exists() and not final.exists():
        raise SystemExit(f"no checkpoint for {args.tag}: looked for {best.name}")

    if final.exists():
        ck = final
    else:
        blob = torch.load(best, map_location="cpu")
        state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
        ck = BENCH / f"{stem}.rescued.pt"
        torch.save(state, ck)
        print(f"unwrapped {best.name} -> {ck.name}")

    dst = BENCH / f"{stem}.json"
    if dst.exists():
        raise SystemExit(f"{dst.name} already exists -- the run wrote its own record, "
                         f"and overwriting it would discard the history and test "
                         f"sections this script cannot reconstruct.")
    dst.write_text(json.dumps({
        "run": stem,
        "model": args.model,
        "tag": args.tag,
        "rescued": True,
        "args": {"model": args.model, "classes": args.classes, "prep": args.prep,
                 "resize": bool(args.resize), "camera_profile": args.camera_profile},
        "checkpoint": str(ck),
    }, indent=1))
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
