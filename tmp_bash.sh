#!/usr/bin/env bash
for prompt in {1..8}
    do
        python DataValueEstimation_DVRL_v2.py  --input_seq word --target_prompt_id ${prompt}
    done