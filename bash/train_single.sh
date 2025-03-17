modality=$1

python train_single.py \
Datasets/Fire \
--dataset=fire_aligned_${modality} \
--model=efficientdetv2_dt \
--batch-size=8 \
--amp \
--lr=1e-3 \
--opt adam \
--sched plateau \
--num-classes=5 \
--save-images \
--workers=8 \
--mean 0.29985648 0.30955142 0.31048897 \
--std 0.21458748 0.20659873 0.21154478