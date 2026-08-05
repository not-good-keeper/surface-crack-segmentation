set -e
P=.venv/Scripts/python.exe
for SD in 11 22 33; do
  for V in smpslim_timm-mobilenetv3_large_100 smp_unet_timm-mobilenetv3_small_100; do
    $P bench/train.py --model $V --tag mseed$SD --seed $SD --epochs 5 --train-steps 250 \
       --batch 48 --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
  done
done
