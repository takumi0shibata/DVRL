#!/usr/bin/env bash
for prompt in {1..8}
do
    for seed in 12 22 32 42 52
    do
        python train_PAES.py --test_prompt_id ${prompt} --device linux --seed ${seed}
    done
done