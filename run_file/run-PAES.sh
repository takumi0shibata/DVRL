#!/usr/bin/env bash
pjname=("DVRL" "LOO" "DataShapley")
batch_size=10
epochs=50
run_name="train-PAES"
valuation_method=("DVRL-pos" "LOO-pos" "DataShapley-pos")
seed=12
device="cuda"

for ((i=0; i<${#pjname[@]}; i++))
do
    for prompt in {1..8}
    do
        python train_PAES.py \
            --wandb \
            --pjname ${pjname[i]} \
            --run_name ${run_name} \
            --valuation_method ${valuation_method[i]} \
            --target_prompt_id ${prompt} \
            --seed ${seed} \
            --batch_size ${batch_size} \
            --epochs ${epochs} \
            --device ${device}
    done
done