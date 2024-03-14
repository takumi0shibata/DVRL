#!/usr/bin/env bash
for prompt in {1..8}
do
    python train_DVRL_DomainAdaptation.py --test_prompt_id ${prompt} --experiment_name DVRL_DomainAdaptation${prompt}_devsize30 --metric qwk --dev_size 30
    # python visualize_DomainAdaptation.py --test_prompt_id ${prompt}
done