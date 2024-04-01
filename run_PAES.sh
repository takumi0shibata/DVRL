#!/usr/bin/env bash
for prompt in {1..8}
do
    for seed in 12
    do
        # python train_PAES.py --test_prompt_id ${prompt} --seed ${seed}
        # python train_PAES_usedev.py --test_prompt_id ${prompt} --seed ${seed}
        python DVRL-PAES.py --test_prompt_id ${prompt} --seed ${seed} --model_type normal --epochs 50 --batch_size 10
    done
done