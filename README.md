# DVRL for AES Project

Welcome to the DVRL_on_torch project! This project is primarily focused on training the DVRL (Data Valuation using Reinforcement Learning) model using PyTorch. The main objective is to provide a robust framework for evaluating the value of data in various machine learning tasks, thereby enhancing model performance through data valuation.

## Getting Started

To embark on this project, ensure Python is installed on your system. You will also need to install a number of dependencies detailed in the `requirements.txt` file.

### Installation

1. Clone this repository to your local machine.
2. To install the necessary dependencies, execute `pip install -r requirements.txt` in your terminal.

### Usage

To commence training the DVRL model, execute the `train_dvrl.py` script. This script is designed to accept multiple command-line arguments, allowing for a tailored training experience. Below is an example command:
```
python train_DVRL_DataValueEstimate.py --test_prompt_id 1 --seed 12
```

## 現状わかっていることまとめ
### 2024/02/16
プロンプト２，４，７以外で適切にノイズ検出できることが判明（シード１２，Reward MSE, 移動平均によるベースラインあり）
### 2024/02/18
プロンプト４について
- 移動平均によるベースラインを，定数によるベースラインに変更→効果なし
- さらにRewardをQWKに変更→適切にノイズ検出ができるように
プロンプト２，７について
- 移動平均によるベースラインを，定数によるベースラインに変更→効果なし
- さらにRewardをQWKに変更→効果なし
- さらにシードを４２に変更→適切にノイズ検出ができるように
プロンプト２，４，７全体について
- うまくノイズ検出ができないときの結果を分析すると，逆張りで収束していることがわかった．つまり，データの価値が高い順にノイズ判定していくと，適切にノイズ検出プロットが描ける
現在の課題
- DVRLの収束するしないが初期値に依存する（データの価値が真逆になるように収束してしまう）問題の解決
