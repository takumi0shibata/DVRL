#!/usr/bin/env bash

# Define the project name
PROJECT_NAME="DVRL-Llama2"

for prompt in {2..8}
do
    for top_p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        python Llama2-DVRL.py --test_prompt_id ${prompt} --wandb --top_p ${top_p} --pjname ${PROJECT_NAME}
    done
    
    for seed in 12
    do  
        python Llama2-DevOnly.py --test_prompt_id ${prompt} --seed ${seed} --wandb --pjname ${PROJECT_NAME}
        python Llama2-FullSource.py --test_prompt_id ${prompt} --seed ${seed} --wandb --pjname ${PROJECT_NAME}
    done
done