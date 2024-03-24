#!/usr/bin/env bash
for prompt in {1..8}
do
    for seed in 12 22 32 42 52
    do  
        python train_MLP.py --test_prompt_id ${prompt} --experiment_name DVRL_DomainAdaptation --seed ${seed}
        python train_BERT_fullsource.py --test_prompt_id ${prompt} --experiment_name DVRL_DomainAdaptation --seed ${seed}
    done
done