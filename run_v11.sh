#!/usr/bin/env bash
# Resize instead of crop. This is a correctness fix, not a tuning change.
#
# Every earlier run trained and scored on 256x256 crops taken at native scale and
# centred on the defect. app/inference.py resized the whole frame. Both hand the model
# a (3,256,256) tensor, so no metric could see the difference, and the measured numbers
# described an input the product never produces:
#
#   test_factory, V9_22   crop   clDice 0.7196  detection 0.949
#                         resize clDice 0.2556  detection 0.423
#
# Worse than the scale gap: a crack-centred crop never shows the model the rest of the
# part. Applied to a whole surface it painted 9.8 % of pixels as defect against 0.69 %
# ground truth -- 14x over-prediction, with 94 % of it nowhere near a real defect.
#
# Training on the whole frame fixes both: the model sees the clean majority of every
# part, and the measured path becomes the deployed path (T-02).
#
# 40 epochs is the wall clock left before the deadline, with time held back for
# evaluation and export. Everything else is held at the v10 values.
set -u
M=smpslim_timm-mobilenetv3_small_100
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --epochs 40 --batch 48 --train-steps 600 --lr 3e-4 --cosine \
  --resize --prep bilateral \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.25 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V11_22
