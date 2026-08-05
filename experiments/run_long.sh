set -e
P=.venv/Scripts/python.exe
$P bench/train.py --model smp_unet_efficientnet-b0 --tag L6_ours --seed 1337 --epochs 6 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth --synth-frac 0.45 --camera-aug
$P bench/train.py --model smp_unet_efficientnet-b0 --tag L6_theirs --seed 1337 --epochs 6 --train-steps 250 --batch 48 --eval-batches 40 --synth data/df_patches --synth-frac 0.45 --camera-aug
$P bench/train.py --model smp_unet_efficientnet-b0 --tag L6_combined --seed 1337 --epochs 6 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth,data/df_bal --synth-frac 0.45 --camera-aug
