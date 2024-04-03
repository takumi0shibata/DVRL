#!/usr/bin/env bash
for prompt in {1..8}
do
    python PAES-DVRL.py --test_prompt_id ${prompt} --model_type normal --epochs 50 --batch_size 10
    for seed in 12 22 32 42 52
    do
        python PAES-FullSource.py --test_prompt_id ${prompt} --seed ${seed} --model_type normal --epochs 50 --batch_size 10
    done
done