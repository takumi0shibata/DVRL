#!/usr/bin/env bash
pjname=("DVRL" "LOO" "DataShapley")
batch_size=512
epochs=100
run_name="train-MLP"
valuation_method_word=("DVRL-word" "LOO-word" "DataShapley-word")
valuation_method_pos=("DVRL-pos" "LOO-pos" "DataShapley-pos")
seed=12
device="cuda"

for ((i=0; i<${#pjname[@]}; i++))
do
    for prompt in {1..8}
    do
        python train_MLP.py \
            --wandb \
            --pjname ${pjname[i]} \
            --run_name ${run_name} \
            --valuation_method ${valuation_method_word[i]} \
            --target_prompt_id ${prompt} \
            --seed ${seed} \
            --batch_size ${batch_size} \
            --epochs ${epochs} \
            --device ${device}
        
        python train_MLP.py \
            --wandb \
            --pjname ${pjname[i]} \
            --run_name ${run_name} \
            --valuation_method ${valuation_method_pos[i]} \
            --target_prompt_id ${prompt} \
            --seed ${seed} \
            --batch_size ${batch_size} \
            --epochs ${epochs} \
            --device ${device}
    done
done