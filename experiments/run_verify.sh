set -e
P=.venv/Scripts/python.exe
$P bench/train.py --model smpslim_timm-mobilenetv3_large_100 --tag v2 --seed 777 --epochs 4 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
$P bench/train.py --model smp_unet_timm-mobilenetv3_small_100 --tag v2 --seed 777 --epochs 4 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
$P bench/train.py --model smpslim_timm-mobilenetv3_large_100 --tag FINAL --seed 2026 --epochs 4 --train-steps 300 --batch 48 --eval-batches 60 --synth data/synth --synth-frac 0.45 --camera-aug --save
