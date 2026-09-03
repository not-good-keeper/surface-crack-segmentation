#!/usr/bin/env bash
# V15 = V12's recipe with one change: a factory-material sampling quota.
#
# The training pool is ~70 % asphalt and plaster -- civil crack sets whose defects are
# large -- while the headline split is entirely fine factory defects. The distributions
# are far apart and the model's scores follow the gap exactly:
#
#   median defect area   train 0.0207   test_factory 0.0055
#   share under 0.005    train 13.1 %   test_factory 47.4 %
#   model clDice         above that line 0.6315   below it 0.5285
#
# Per material the same relationship appears unprompted: plastic has 90 % small-defect
# training images and is the best-scoring material (0.681); steel has 6 % and is the
# worst of the seen ones (0.527).
#
# Keyed on material rather than directly on defect size, deliberately. Restricted to
# factory materials, train (40.0 % small) and val (39.9 %) already look like
# test_factory (46.2 %) -- so the size gap is a material-composition effect, and
# sampling on size would be fitting the training mix to a test-set statistic. Material
# is justified by ADR-002 (factory-first, civil auxiliary) and confirmed on val.
#
# BGR, matching V12: V14 established that RGB training is worse on the headline at every
# threshold, even though serving BGR weights as RGB was a real -0.1412 deployment bug.
set -u
M=smpslim_timm-mobilenetv3_small_100
EPOCHS=${EPOCHS:-36}
FF=${FF:-0.5}
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --epochs "$EPOCHS" --batch 48 --train-steps 600 --lr 3e-4 --cosine \
  --resize --prep bilateral --factory-frac "$FF" \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.15 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V15_FAC
