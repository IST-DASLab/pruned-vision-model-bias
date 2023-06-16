#!/usr/bin/env bash

declare -a manual_seed=(23 24 25)

for ((j=0;j<${#manual_seed[@]};++j));
do

python main.py \
	--dset=celeba \
	--dset_path=/home/Datasets/ \
	--arch=mobilenet \
	--config_path=./configs/celeba_mobilenet_dense_ep100.yaml \
	--workers=4 \
	--epochs=100 \
	--batch_size=256 \
	--gpus=$1 \
	--manual_seed=${manual_seed[j]} \
	--use_wandb \
	--experiment_root_path "./experiments_celeba_mobilenet_bias" \
	--exp_name=mobilenet_s0 \
	--wandb_project "celeba_mobilenet_bias" 

done
