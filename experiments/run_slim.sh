set -e
P=.venv/Scripts/python.exe
for V in smpslim_timm-mobilenetv3_small_100 smpslim_mobilenet_v2 smpslim_timm-mobilenetv3_large_100; do
  $P bench/train.py --model $V --tag ts --epochs 4 --train-steps 250 --batch 48 \
     --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
done
