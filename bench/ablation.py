"""Does synthetic data help on a material with no real labels?

    A  abl_nowood      wood excluded entirely
    B  abl_woodsynth   synthetic wood included; still no real wood

Neither arm ever sees a real wood crack, so B - A on test_unseen_material is what the
generator is worth. The premise is the field one: you can always photograph a clean
surface of a new material, but you cannot get annotated cracks on it.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/Scripts/python.exe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="winning architecture")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=700)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--synth", default="data/synth")
    ap.add_argument("--synth-frac", type=float, default=0.45)
    ap.add_argument("--eval-batches", type=int, default=80)
    args = ap.parse_args()

    arms = [
        ("abl_nowood", ["--exclude-materials", "wood"]),
        ("abl_woodsynth", []),
    ]
    for tag, extra in arms:
        print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
        cmd = [PY, "bench/train.py", "--model", args.model, "--tag", tag,
               "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--train-steps", str(args.train_steps),
               "--eval-batches", str(args.eval_batches),
               "--synth", args.synth, "--synth-frac", str(args.synth_frac)] + extra
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"{tag} FAILED")
            return 1

    subprocess.run([PY, "bench/recommend.py"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
