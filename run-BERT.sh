#!/usr/bin/env bash
for prompt in {1..8}
do
    python BERT-DVRL.py --test_prompt_id ${prompt} --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased
    for seed in 12 22 32 42 52
    do  
        python BERT-DevOnly.py --test_prompt_id ${prompt} --seed ${seed} --max_length 512 --batch_size 16 --epochs 30 --model_name bert-base-uncased
        python BERT-FullSource.py --test_prompt_id ${prompt} --seed ${seed} --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased
    done
done