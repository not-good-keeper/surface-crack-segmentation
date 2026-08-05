set -e
P=.venv/Scripts/python.exe
for V in tinyseg_w8_d4 tinyseg_w12_d4 tinyseg_w16_d4 tinyseg_w16_d5 tinyseg_w24_d4 smp_unet_timm-mobilenetv3_small_100; do
  $P bench/train.py --model $V --tag ts1 --epochs 2 --train-steps 250 --batch 48 \
     --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
done
