#!/usr/bin/env bash

declare -a seed=(8 15 23 30 35)
declare -a attrs=(blond smiling necktie attractive oval-face big-nose)
declare -a labels=(9 31 38 2 25 7)
declare -a negfrac=(0.05 0.05 0.05 0.05 0.35 0.35)
declare -a posfrac=(0.95 0.95 0.95 0.95 0.65 0.65)

declare -a backdoors=(grayscale yellow_square)

for ((i=0;i<${#seed[@]};++i));
do
for ((j=0;j<${#attrs[@]};++j));
do
for ((k=0;k<${#backdoors[@]};++k));
do

echo ${seed[i]} ${attrs[j]} ${labels[j]} ${backdoors[k]}

python main.py \
	--dset=backdoorceleba \
	--dset_path=/home/Datasets/ \
	--arch=resnet18 \
	--config_path=./configs/celeba_resnet18_dense_ep20.yaml \
	--workers=4 \
	--backdoor_type_train=${backdoors[k]} \
	--backdoor_type_test=${backdoors[k]} \
	--backdoor_label=${labels[j]} \
	--backdoor_fracs_train ${negfrac[j]} ${posfrac[j]} \
	--backdoor_fracs_test 0.5 0.5 \
	--epochs=20 \
	--batch_size=256 \
	--gpus=$1 \
	--manual_seed=${seed[i]} \
	--label_indices=${labels[j]} \
	--experiment_root_path "./experiments_celeba_rn18_bias" \
	--exp_name=dense_${backdoors[k]}_${attrs[j]} \
	--wandb_project "celeba_rn18_backdoor_bias_single_label" --use_wandb


done
done 
done

