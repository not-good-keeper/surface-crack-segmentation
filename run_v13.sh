#!/usr/bin/env bash
# 512 input. The hypothesis is that 256 destroys the defect before the model ever sees it.
#
# Test frames have a median long side of 793 px and a p90 of 1262. Scaling those to 256
# is a 3.1-4.9x reduction, which takes a 4 px crack to 0.8-1.3 px -- at or below the
# sampling limit. No amount of training recovers a structure thinner than a pixel, and
# that is the most likely reason clDice on the honest path sits near 0.60 while the same
# weights scored 0.72 on native-scale crops. 512 halves the loss; it does not remove it.
#
#   --init      Warm start from the converged 256 model rather than ImageNet. The network
#               is fully convolutional, so the weights load unchanged and only the scale
#               has to be relearned -- a far smaller thing than relearning the task, and
#               the difference between a useful run and one that is still warming up when
#               the clock runs out.
#
#   --lr 1e-4   A third of the 256 run's peak. Fine-tuning converged weights at the
#               original learning rate discards what the warm start was for.
#
#   --batch 24  Measured, not guessed: 512 costs 0.15 GB/sample, so batch 48 fits in the
#               12 GB card easily. The limit here is wall clock, not memory -- 48 would
#               be a straight 4x per epoch (~900 s). 24 halves that to ~460 s and buys
#               twice as many annealing steps in the time available.
#
# Nothing else moves from run_v12.sh, so a difference in the result is attributable to
# the input size. One caveat: `bilateral` was selected by the sweep at 256, and a spatial
# filter's kernel is scale-dependent -- it may not be the right prep at 512. Held fixed
# here for comparability, and worth re-sweeping if this run wins.
set -u
M=smpslim_timm-mobilenetv3_small_100
EPOCHS=${EPOCHS:-24}
PREP=${PREP:-bilateral}
INIT=${INIT:-data/bench/smpslim_timm-mobilenetv3_small_100_V12_22.best.pt}
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --size 512 --init "$INIT" \
  --epochs "$EPOCHS" --batch 24 --train-steps 600 --lr 1e-4 --cosine \
  --resize --prep "$PREP" \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.15 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V13_512
