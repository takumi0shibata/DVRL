#!/usr/bin/env bash
for prompt in {1..8}
do
    python train_DVRL_DataValueEstimate.py --test_prompt_id ${prompt} --experiment_name DVRL_DataValueEstimate${prompt}
done