set -e
P=.venv/Scripts/python.exe
for SD in 11 22 33; do
  $P bench/train.py --model smpslim_timm-mobilenetv3_small_100 --tag V2_$SD --seed $SD \
     --epochs 3 --train-steps 250 --batch 48 --eval-batches 60 \
     --synth data/synth_core,data/synth_fix --synth-frac 0.45 --camera-aug --ema 0.995
done
