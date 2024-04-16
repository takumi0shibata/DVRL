#!/usr/bin/env bash
for prompt in {1..8}
do

    for top_p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        python Llama2-DVRL.py --test_prompt_id ${prompt} --wandb --top_p ${top_p} --ascending --num_epochs 10
        python Llama2-DVRL.py --test_prompt_id ${prompt} --wandb --top_p ${top_p} --num_epochs 10
    done
    
    for seed in 12 22 32 42 52
    do  
        python Llama2-DevOnly.py --test_prompt_id ${prompt} --seed ${seed} --num_epochs 10 --wandb
        python Llama2-FullSource.py --test_prompt_id ${prompt} --seed ${seed} --num_epochs 10 --wandb
    done
done