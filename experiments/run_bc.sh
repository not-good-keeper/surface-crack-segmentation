set -e
P=.venv/Scripts/python.exe
$P bench/train.py --model smp_unet_efficientnet-b0 --tag ds_B_combined --epochs 2 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth,data/df_patches --synth-frac 0.45 --camera-aug
$P bench/train.py --model smp_unet_efficientnet-b0 --tag ds_C_theirs --epochs 2 --train-steps 250 --batch 48 --eval-batches 40 --synth data/df_patches --synth-frac 0.45 --camera-aug
