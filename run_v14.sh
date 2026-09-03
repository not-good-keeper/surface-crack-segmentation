#!/usr/bin/env bash
# V14 = V12's recipe with exactly one change: --rgb.
#
# The backbone is ImageNet-pretrained and its normalisation constants are defined for
# RGB, but bench/data.py fed it cv2's BGR. Every run up to V12 therefore trained a
# pretrained stem on transposed channels. Serving it correctly cost -0.1412 clDice at
# deployment (fixed separately); this run asks the other half of the question -- what
# the swap cost during training, where it degraded transfer from the pretrained weights
# rather than just the input.
#
# Nothing else moves from run_v12.sh, so any difference is attributable to channel order.
set -u
M=smpslim_timm-mobilenetv3_small_100
EPOCHS=${EPOCHS:-36}
PREP=${PREP:-bilateral}
./.venv/Scripts/python.exe bench/train.py --model $M --classes 3 \
  --epochs "$EPOCHS" --batch 48 --train-steps 600 --lr 3e-4 --cosine \
  --resize --rgb --prep "$PREP" \
  --synth 'data/synth:crack|negative,data/df_patches2:scratch' \
  --synth-frac 0.15 --scratch-frac 0.35 \
  --camera-aug --camera-profile conveyor \
  --ema 0.995 --eval-batches 40 --seed 22 --save --tag V14_RGB
