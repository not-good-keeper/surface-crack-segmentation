set -e
P=.venv/Scripts/python.exe
# G: ours + ONLY their physically-rendered hard negatives (no domain-diluting synthetic cracks)
$P bench/train.py --model smp_unet_efficientnet-b0 --tag G_ours_theirsneg --epochs 2 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth,data/df_neg --synth-frac 0.45 --camera-aug
for V in tinyseg_w12_d4 tinyseg_w16_d4 tinyseg_w16_d5 tinyseg_w24_d4 tinyseg_w8_d5 smp_unet_timm-mobilenetv3_small_100; do
  $P bench/train.py --model $V --tag ts --epochs 4 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
done
