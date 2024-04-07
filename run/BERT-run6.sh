#!/usr/bin/env bash
python BERT-DVRL.py --test_prompt_id 6 --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased --device cuda:5
for seed in 12 22 32 42 52
do  
    python BERT-DevOnly.py --test_prompt_id 6 --seed ${seed} --max_length 512 --batch_size 16 --epochs 15 --model_name bert-base-uncased --device cuda:5
    python BERT-FullSource.py --test_prompt_id 6 --seed ${seed} --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased --device cuda:5
done