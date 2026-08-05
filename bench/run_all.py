"""Run every candidate under identical conditions, then recommend.

Sequential on purpose. Two models training concurrently on one GPU distort both the
epoch times and the CPU-latency measurement, and latency is one of the numbers the
recommendation turns on -- a benchmark that measures contention instead of
architecture is worse than no benchmark.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/Scripts/python.exe")

ORDER = [
    "tiny_unet",
    "smp_unet_timm-mobilenetv3_small_100",
    "smp_unet_efficientnet-b0",
    "smp_dlv3p_timm-mobilenetv3_large_100",
    "lraspp_mnv3",
    "segformer_b0",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--synth", default="data/synth")
    ap.add_argument("--synth-frac", type=float, default=0.45)
    ap.add_argument("--eval-batches", type=int, default=60)
    args = ap.parse_args()

    models = ORDER
    print(f"benchmarking {len(models)} models, {args.epochs} epochs each\n")
    t0 = time.time()
    ok, failed = [], []
    for i, m in enumerate(models, 1):
        print(f"\n{'='*70}\n[{i}/{len(models)}] {m}\n{'='*70}", flush=True)
        cmd = [PY, "bench/train.py", "--model", m,
               "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--train-steps", str(args.train_steps),
               "--eval-batches", str(args.eval_batches)]
        if args.synth:
            cmd += ["--synth", args.synth, "--synth-frac", str(args.synth_frac)]
        r = subprocess.run(cmd, cwd=ROOT)
        (ok if r.returncode == 0 else failed).append(m)
        print(f"[{i}/{len(models)}] {'ok' if r.returncode == 0 else 'FAILED'} "
              f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)

    print(f"\n{'='*70}\n{len(ok)} ok, {len(failed)} failed "
          f"in {(time.time()-t0)/60:.0f} min")
    for m in failed:
        print(f"  FAILED: {m}")
    subprocess.run([PY, "bench/recommend.py"], cwd=ROOT)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
