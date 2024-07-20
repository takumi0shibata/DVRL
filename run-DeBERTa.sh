#!/usr/bin/env bash
pjname=("DVRL" "LOO" "Data Shapley")
max_length=256
batch_size=8
epochs=5
lr=2e-5
model_name="microsoft/deberta-v3-large"
run_name="train-DeBERTa"
valuation_method=("DVRL-word" "LOO-word" "DataShapley-word")
seed=12
device="cuda"

for ((i=0; i<${#pjname[@]}; i++))
do
    for prompt in {1..8}
    do
        python train_Transformers.py \
            --wandb \
            --pjname ${pjname[i]} \
            --run_name ${run_name} \
            --valuation_method ${valuation_method[i]} \
            --target_prompt_id ${prompt} \
            --seed ${seed} \
            --max_length ${max_length} \
            --batch_size ${batch_size} \
            --epochs ${epochs} \
            --model_name ${model_name} \
            --lr ${lr} \
            --device ${device}
    done
done