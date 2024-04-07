#!/usr/bin/env bash

# 共通のパラメータを変数に格納
model_name="microsoft/deberta-v3-large"
test_prompt_id=2
max_length=256
batch_size=8
epochs=4
# test_prompt_idの値に応じてdeviceを設定
device="cuda:$((test_prompt_id - 1))"
lr=2e-5

# BERT-DVRL.pyの実行
python BERT-DVRL.py \
  --test_prompt_id $test_prompt_id \
  --max_length $max_length \
  --batch_size $batch_size \
  --epochs $epochs \
  --model_name $model_name \
  --device $device \
  --lr $lr \
  --run_name "debertav3large-DVRL-nodev"

# seedの値を配列に格納
seeds=(12 22 32 42 52)

# seedの値ごとにBERT-DevOnly.pyとBERT-FullSource.pyを実行
for seed in "${seeds[@]}"
do
  python BERT-DevOnly.py \
    --test_prompt_id $test_prompt_id \
    --seed $seed \
    --max_length $max_length \
    --batch_size $batch_size \
    --epochs $epochs \
    --model_name $model_name \
    --device $device \
    --lr $lr \
    --run_name "debertav3large-DevOnly-nodev"

  python BERT-FullSource.py \
    --test_prompt_id $test_prompt_id \
    --seed $seed \
    --max_length $max_length \
    --batch_size $batch_size \
    --epochs $epochs \
    --model_name $model_name \
    --device $device \
    --lr $lr \
    --run_name "debertav3large-FullSource-nodev"
done