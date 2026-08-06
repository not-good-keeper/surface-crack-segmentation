#!/usr/bin/env bash
# Final run. Whole-frame resize, matching what app/inference.py feeds the model.
#
# Changes from v11, both from faults found while v11 was in flight:
#
#   --synth-frac 0.15   Synthetic patches are 256x256 composites built at native crop
#                       scale, and the resize branch only touches real rows. So real
#                       cracks are now ~1.56x thinner (400 -> 256) while synthetic ones
#                       keep their original width. Cutting their share reduces how much
#                       of each batch teaches a crack width the test set does not have.
#                       Judgement call under time pressure, not a measured result.
#
#   --prep              Set from bench/prep_sweep.py run against the v11 epoch-5
#                       checkpoint under the resize regime.
#
# EPOCHS is sized to land before the deadline at the measured epoch time, not chosen.
set -u
M=smpslim_timm-mobilenetv3_small_100
EPOCHS=${EPOCHS:-42}
PREP=${PREP:-bilateral}
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --epochs "$EPOCHS" --batch 48 --train-steps 600 --lr 3e-4 --cosine \
  --resize --prep "$PREP" \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.15 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V12_22
