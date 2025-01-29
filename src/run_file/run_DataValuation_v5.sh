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

# loss_lambda のリスト
lambda_list=("1.0" "0.5" "0.0")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for lambda in "${lambda_list[@]}"
  do
    echo "Running with pred_model: ${pred_model}, lambda: ${lambda}"

    python src/DataValueEstimation_DVRL_v5.py \
        --wandb \
        --pjname "DVRL-V5" \
        --target_prompt_id "${prompt}" \
        --seed 22 \
        --device "${device}" \
        --pred_model "${pred_model}" \
        --loss_lambda "${lambda}" \
        --ot
  done
done