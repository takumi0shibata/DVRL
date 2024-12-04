#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

# Promptをコンソールから入力を受け付ける
read -p "Enter the target prompt ID: " prompt

# deviceをcuda:{prompt}に設定
device="cuda:$((prompt))"

# pred_model のリスト
# pred_models=("mlp" "features_model")
pred_models=("features_model")

# n_clusters のリスト
n_clusters_list=("10000" "2000" "1000")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for n_clusters in "${n_clusters_list[@]}"
  do
    echo "Running with pred_model: ${pred_model}, n_clusters: ${n_clusters}"

    python src/DataValueEstimation_DVRL_v4.py \
        --wandb \
        --pjname "DVRL-V4" \
        --target_prompt_id "${prompt}" \
        --seed 12 \
        --device "${device}" \
        --pred_model "${pred_model}" \
        --n_clusters "${n_clusters}"
  done
done