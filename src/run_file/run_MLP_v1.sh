#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

# Promptをコンソールから入力を受け付ける
read -p "Enter the target prompt ID: " prompt

# deviceをcuda:{prompt}に設定
device="cuda:$((prompt))"

# pred_model のリスト
pred_models=("mlp" "features_model")
# pred_models=("features_model")

# n_clusters のリスト
dev_size_list=("100" "500" "1000")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for dev_size in "${dev_size_list[@]}"
  do
    echo "Running with pred_model: ${pred_model}, dev_size: ${dev_size}"

    python src/train_models/train_MLP_v1.py \
        --wandb \
        --pjname "DVRL-V1" \
        --target_prompt_id "${prompt}" \
        --seed 12 \
        --device "${device}" \
        --pred_model "${pred_model}" \
        --dev_size "${dev_size}"
  done
done