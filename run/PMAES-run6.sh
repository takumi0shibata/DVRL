#!/usr/bin/env bash

# 共通のパラメータを変数に格納
target_id=6
# test_prompt_idの値に応じてdeviceを設定
device="cuda:$((target_id - 1))"

for seed in 12 22 32 42 52
do
	python PMAES-FullSource.py --seed $seed --target_id $target_id --device $device
done