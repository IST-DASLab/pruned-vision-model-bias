#!/usr/bin/env bash
declare -a manual_seed=(23 24 25)

for ((j=0;j<${#manual_seed[@]};++j));
do

python main.py \
	--dset=celeba \
	--dset_path=/home/Datasets/ \
	--arch=resnet18 \
	--config_path=./configs/celeba_resnet18_dense_ep100.yaml \
	--workers=4 \
	--epochs=100 \
	--batch_size=256 \
	--gpus=$1 \
	--manual_seed=${manual_seed[j]} \
	--use_wandb \
	--experiment_root_path "./experiments_celeba_rn18_bias" \
	--exp_name=resnet18_s0
	--wandb_project "celeba_rn18_bias" 

done
