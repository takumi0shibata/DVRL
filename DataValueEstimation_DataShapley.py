import os
import torch.nn as nn
import torch
import numpy as np
import argparse
import torch
import wandb
from collections import deque

from utils.general_utils import set_seed
from utils.load_data import load_data_DVRL, load_data_PAES
from dvrl.predictor import MLP
from models.features import FeaturesModel

# Data Shapley 値計算関数
def data_shapley(X_train, y_train, X_test, y_test,  input_seq, max_iter=1000, threshold=0.05):
    n_samples = X_train.shape[0]
    input_dim = X_train.shape[1]
    shapley_values = np.zeros(n_samples)
    past_shapley_values = deque(maxlen=100)
    t = 0
    while t < max_iter:
        t += 1
        permutation = np.random.permutation(n_samples)
        if 'word' in input_seq:
            model = MLP(input_dim)
        elif 'pos' in input_seq:
            model = FeaturesModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            init_loss = criterion(model(X_test_tensor).squeeze(), y_test_tensor).item()
        losses.append(init_loss)

        for j in range(n_samples):
            X_train_point = torch.tensor(X_train[permutation[j]].reshape(1, -1), dtype=torch.float32)
            y_train_point = torch.tensor(y_train[permutation[j]].reshape(1, -1), dtype=torch.float32)

            # Fit
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_point)
            loss = criterion(y_train_point, y_pred)
            loss.backward()
            optimizer.step()

            # Eval
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor)
                loss = criterion(y_test_tensor, y_pred.squeeze()).item()

            # Update Shapley value
            shapley_values[permutation[j]] = (t - 1) / t * shapley_values[permutation[j]] + 1 / t * (loss - losses[-1])
            losses.append(loss)

        # Early stopping check (after the initial 100 iterations)
        if t > 100:
            # 100イテレーション前の Shapley 値を取得
            past_shapley_value = past_shapley_values.popleft()
            convergence_criteria = np.mean(np.abs(shapley_values - past_shapley_value) / np.abs(shapley_values))
            print(f"Iteration {t}: {convergence_criteria}")
            if convergence_criteria < threshold:
                print(f"Early stopping at iteration {t}")
                return shapley_values
        else:
            print(f"Iteration {t}")

        past_shapley_values.append(shapley_values.copy())  # 現在の Shapley 値をキューに追加

    return shapley_values


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    ###################################################
    # Step1. Create/Load Text Embedding
    ###################################################
    print('Loading data...')
    if args.input_seq == 'word':
        dvrl_data = load_data_DVRL(
            f'data/cross_prompt_attributes/{target_prompt_id}/',
            args.attribute_name,
            args.embedding_model,
            device,
            devsize=args.dev_size
        )
    elif args.input_seq == 'pos':
        dvrl_data = load_data_PAES(
            f'data/cross_prompt_attributes/{target_prompt_id}/',
            args.attribute_name,
            args.embedding_model,
            device,
            devsize=args.dev_size
        )
        dvrl_data['x_source'] = dvrl_data['x_source'][1]
        dvrl_data['x_dev'] = dvrl_data['x_dev'][1]

    ###################################################
    # Step2. Training LOO
    ###################################################
    if args.wandb:
        wandb.init(
            project=args.wandb_pjname,
            name=args.experiment_name + f'_{target_prompt_id}_{args.input_seq}',
            config=dict(args._get_kwargs())
        )
    
    ds_scores = data_shapley(
        dvrl_data['x_source'],
        dvrl_data['y_source'],
        dvrl_data['x_dev'],
        dvrl_data['y_dev'],
        args.input_seq,
        max_iter=1000,
        threshold=0.05,
    )
    
    # Save the leave-one-out scores
    os.makedirs(f'outputs/Estimated_Data_Values/DataShapley-{args.input_seq}', exist_ok=True)
    np.save(f'outputs/Estimated_Data_Values/DataShapley-{args.input_seq}/estimated_data_value{target_prompt_id}.npy', np.array(ds_scores))
    print('DataShapley scores saved.')
    print(ds_scores)

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Data Shapley")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_pjname', type=str, default='テスト')
    parser.add_argument('--experiment_name', type=str, default='DataShapley_DataValueEstimation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--input_seq', type=str, default='word', choices=['word', 'pos'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)
