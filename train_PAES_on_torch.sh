#!/usr/bin/env bash
for seed in 12
do
    for prompt in {3..8}
    do
        python train_PAES_on_torch.py --test_prompt_id ${prompt} --seed ${seed} --device linux
    done
done