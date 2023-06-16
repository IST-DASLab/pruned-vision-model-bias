#!/usr/bin/env bash

declare -a seed=(15 17 19 21 23)
declare -a attrs=(blond smiling oval-face big-nose mustache receding-hairline bags-under-eyes)
declare -a labels=(9 31 25 7 22 28 3)
declare -a arch=(resnet18)
declare -a sparsities=(80 90 95 98 99 995)


for ((i=0;i<${#seed[@]};++i));
do
for ((j=0;j<${#attrs[@]};++j));
do
for ((l=0;l<${#sparsities[@]};++l));
do

echo ${seed[i]} ${attrs[j]} ${labels[j]} ${sparsities[l]}

python main.py \
	--dset=celeba \
	--dset_path=/home/Datasets/ \
	--arch=${arch} \
	--config_path=./configs/celeba_${arch}_gmp${sparsities[l]}_ep20.yaml \
	--workers=4 \
	--epochs=20 \
	--batch_size=256 \
	--gpus=$1 \
	--manual_seed=${seed[i]} \
	--label_indices=${labels[j]} \
	--experiment_root_path "./experiments_celeba_rn18_bias" \
	--exp_name=celeba_gmp_sp${sparsities[l]}_${attrs[j]} \
	--wandb_project "celeba_rn18_bias_single_attr" --use_wandb


done
done 
done

