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
--mean 0.53584253 0.53584253 0.53584253 \
--std 0.24790472 0.24790472 0.24790472