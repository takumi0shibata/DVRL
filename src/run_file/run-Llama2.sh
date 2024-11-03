#!/usr/bin/env bash

# Define the project name
pjname=("DVRL" "LOO" "DataShapley")
valuation_method=("DVRL-word" "LOO-word" "DataShapley-word")
seed=12
run_name="train-Llama2"

for ((i=0; i<${#pjname[@]}; i++))
do
    for prompt in {1..8}
    do
        for top_p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            python train_Llama2.py \
                --target_prompt_id ${prompt} \
                --wandb \
                --run_name ${run_name} \
                --top_p ${top_p} \
                --pjname ${pjname[i]} \
                --valuation_method ${valuation_method[i]} \
                --seed ${seed}
        done
    done
done