#!/usr/bin/env bash
# Splits v7, three classes, conveyor profile. Two arms, run head to head.
#
# Arm A (V8A) is the new recipe. It changes two things at once, deliberately:
#
#   1. Per-source synthetic kinds. `--synth-kinds` filters the concatenated frame, so
#      the previous command admitted DefectForge's 27,295 cracks as well as its
#      scratches -- 30 % of every synthetic crack the model saw, contradicting this
#      script's own comment. Cracks and distractors now come from our compositor,
#      scratches from DefectForge, which is the only synthetic scratch source.
#   2. --synth-frac 0.45 -> 0.25. Real data was outnumbered inside the batch, which
#      made sense when plastic had 0 real crack masks and steel had 300. It no longer
#      does: v7 carries 936 real plastic and 685 real steel masks, so the synthetic
#      pool is no longer the only signal those materials have.
#
# Arm B (V8B) is the OLD recipe on the SAME splits, one seed. Without it the arm-A
# numbers cannot be attributed to anything -- v7 test splits differ from v5 and v6, so
# the stored C3B/C3V6 results are not a valid control. B answers "did this help?"; it
# does not separate change 1 from change 2, and no claim should pretend otherwise.
#
# 4 epochs, not 8. The 8-epoch run selected epoch 1 as best and never beat it in five
# more, while val kept climbing -- the early-peak pattern this headline always shows.
# Best-epoch restoration makes the extra epochs cost time without buying accuracy.
#
# --scratch-frac 0.35 unchanged and still load-bearing: at its natural ~6 % frequency
# the scratch class was never predicted on a single pixel while cracks scored 0.53.
# Rebalancing stays at the sampler, never in the loss -- loss weighting buys scratch
# recall by over-painting, and over-rejection is the failure this system exists to stop.
set -u
M=smpslim_timm-mobilenetv3_small_100
PY=./.venv/Scripts/python.exe
COMMON="--model $M --classes 3 --epochs 4 --batch 48 --train-steps 600 --lr 3e-4
        --camera-aug --camera-profile conveyor --ema 0.995 --eval-batches 40 --save"

# Arm A first at one seed, then the control, then the remaining seeds: the head-to-head
# is readable after two runs instead of four.
$PY bench/train.py $COMMON --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.25 --scratch-frac 0.35 --seed 11 --tag V8A_11

$PY bench/train.py $COMMON --synth data/synth,data/df_patches2 \
  --synth-kinds crack,scratch,negative \
  --synth-frac 0.45 --scratch-frac 0.35 --seed 11 --tag V8B_11

for S in 22 33; do
  $PY bench/train.py $COMMON --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
    --synth-frac 0.25 --scratch-frac 0.35 --seed $S --tag V8A_$S
done
