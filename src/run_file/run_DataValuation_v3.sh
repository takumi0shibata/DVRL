#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
    for input_seq in "word" "pos"
    do
        for prompt in {1..8}
        do
            python src/DataValueEstimation_DVRL_v3.py \
                --wandb \
                --pjname "[TEST]DataValuation-v3" \
                --run_name "train-MLP" \
                --target_prompt_id ${prompt} \
                --seed ${seed} \
                --device "cpu" \
                --input_seq ${input_seq} \
                --device "cpu"
        done
    done
done
