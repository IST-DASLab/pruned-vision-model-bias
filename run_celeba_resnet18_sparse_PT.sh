#!/usr/bin/env bash
declare -a manual_seed=(23 24 25)
declare -a sparsities=(80 90 95 98 99 995)

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((l=0;l<${#sparsities[@]};++l));
do

python main.py \
	--dset=celeba \
	--dset_path=/home/Datasets/ \
	--arch=resnet18 \
	--config_path=configs/celeba_resnet18_postgmps${sparsities[l]}_ep80_adam_lr0001.yaml \
	--from_checkpoint_path="./experiments_celeba_rn18_bias/resnet18_s0/seed22/20221011190139/best_dense_checkpoint.ckpt" \
	--workers=4 \
	--epochs=80 \
	--batch_size=256 \
	--gpus=$1 \
	--manual_seed=${manual_seed[j]} \
	--use_wandb \
	--experiment_root_path "./experiments_bias" \
	--exp_name=resnet18_post_gmps${sparsities[l]} \
	--wandb_project "celeba_rn18_bias" 

done
done
