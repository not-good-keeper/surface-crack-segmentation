#!/usr/bin/env bash
# Final training. One seed at a time, longest-first, best seed first.
#
# Seeds run in descending order of their v8 headline, so if the run is cut short the
# time already spent went to the most promising seed rather than to whichever one
# happened to be first:
#
#     seed 22   test_factory clDice 0.6887   <- best, runs first
#     seed 11                        0.6651
#     seed 33                        0.6775
#
# 20 epochs rather than 4. All three v8 runs reached their best headline at the FINAL
# epoch (ADR-017), so best-epoch restoration was never selecting anything but the last
# one -- the cap was not preventing overfitting, it was stopping early. The cap came
# from the v5/v6 rounds where the headline genuinely peaked at epoch 1-2 and then
# decayed ~0.06 while val kept climbing; that behaviour is absent under v7.
#
# A long run is safe to interrupt. bench/train.py mirrors the best-so-far weights to
# data/bench/<run>.best.pt on every improvement, so killing this at any epoch still
# leaves the best model found on disk. Under-training is the failure we have now
# measured three times; over-running costs only wall clock.
#
# Everything else is held fixed so these runs stay comparable with V8A:
#   per-source synth kinds     cracks/distractors from our compositor, scratches from
#                              DefectForge -- the only synthetic scratch source (ADR-013)
#   --synth-frac 0.25          real data is 75 % of a batch (ADR-014)
#   --scratch-frac 0.35        sampler-level quota, never loss weighting (ADR-012)
#   --camera-profile conveyor  belt-axis blur + specular, no fisheye (ADR-011)
#   --ema 0.995                cut seed variance ~3x
set -u
M=smpslim_timm-mobilenetv3_small_100
PY=./.venv/Scripts/python.exe
for S in 22 33 11; do
  $PY bench/train.py --model $M --classes 3 \
    --epochs 20 --batch 48 --train-steps 600 --lr 3e-4 \
    --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
    --synth-frac 0.25 --scratch-frac 0.35 \
    --camera-aug --camera-profile conveyor \
    --ema 0.995 --eval-batches 40 --seed $S --save --tag V9_$S
done
