#!/usr/bin/env bash
for prompt in {1..8}
do
    python 0_train_DVRL_pos.py --test_prompt_id ${prompt}
    # python train_DVRL_DomainAdaptation_FeatureModel.py --test_prompt_id ${prompt} --experiment_name DVRL_DomainAdaptation_FeatureModel${prompt}_devsize40 --dev_size 40
done