#!/usr/bin/env bash
for prompt in {1..8}
do
    python MLP-DVRL.py --test_prompt_id ${prompt}
    for seed in 12 22 32 42 52
    do
        python MLP-DevOnly.py --test_prompt_id ${prompt} --seed ${seed}
        python MLP-FullSource.py --test_prompt_id ${prompt} --seed ${seed}
    done
done