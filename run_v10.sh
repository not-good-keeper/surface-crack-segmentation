#!/usr/bin/env bash
# Final run. One seed, the full remaining budget, bilateral denoise in the input path.
#
# Two changes over v9, both measured rather than guessed:
#
# --prep bilateral   bench/prep_sweep.py swept 22 transforms on val and reported the
#                    winners on the frozen splits. Every denoiser trades the same way --
#                    detection, crack recall and unseen-material response all rise
#                    together with false positives -- so the choice is which point on
#                    that curve stays inside NFR-03. median3 scored the best headline
#                    (+0.016 clDice) and pushed fp_area to 0.0061, through the 0.005
#                    ceiling. bilateral takes +0.007 clDice, detection 0.949 -> 0.962 and
#                    wood clDice +0.088 at fp_area 0.0040, which is inside it.
#                    CLAHE, the reflex choice for surface inspection, costs 0.057.
#
# --cosine           v9 swung 0.06 headline between adjacent epochs at a constant lr,
#                    which is the optimiser orbiting a minimum rather than converging.
#                    Over 80 epochs that also turns best-epoch selection into a lottery:
#                    80 draws from a noisy series picks the luckiest, not the best.
#                    Decaying to 2 % makes the tail stable and the final epoch honest.
#
# 80 epochs is the wall-clock budget, not a tuned number: ~245 s/epoch against the time
# left before the deadline, with an hour held back for evaluation, export and writeup.
#
# Seed 22 alone -- it led v8 and v9 on the headline. bench/train.py mirrors the
# best-so-far weights to data/bench/<run>.best.pt on every improvement, so a kill at any
# epoch still leaves the best model found on disk.
#
# Everything else is held at the v9 values so the comparison stays clean (ADR-011..014).
set -u
M=smpslim_timm-mobilenetv3_small_100
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --epochs 80 --batch 48 --train-steps 600 --lr 3e-4 --cosine \
  --prep bilateral \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.25 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V10_22
