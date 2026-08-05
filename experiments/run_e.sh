set -e
P=.venv/Scripts/python.exe
$P bench/train.py --model smp_unet_efficientnet-b0 --tag ds_E_ours_theirsbal --epochs 2 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth,data/df_bal --synth-frac 0.45 --camera-aug
$P bench/train.py --model smp_unet_efficientnet-b0 --tag ds_F_ours_theirscracks --epochs 2 --train-steps 250 --batch 48 --eval-batches 40 --synth data/synth,data/df_bal --synth-frac 0.60 --camera-aug
