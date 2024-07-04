"""Training on LOO"""

import os
import torch
import numpy as np
import argparse
import torch
import wandb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from transformers import AutoConfig
from utils.general_utils import set_seed
from utils.load_data import load_data_DVRL, load_data_PAES
from dvrl.predictor import MLP
from models.features import FeaturesModel
from utils.dvrl_utils import fit_func, pred_func


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    batch_size = args.batch_size
    epochs = args.epochs
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
    
    # Create predictor
    print('Creating predictor model...')
    if args.input_seq == 'word':
        config = AutoConfig.from_pretrained(args.embedding_model)
        pred_model = MLP(input_feature=config.hidden_size).to(device)
    elif args.input_seq == 'pos':
        pred_model = FeaturesModel().to(device)
    torch.save(pred_model.state_dict(), f'tmp/init_model{target_prompt_id}.pth')

    # Calculate the leave-one-out scores for source data
    fit_func(
        pred_model,
        dvrl_data['x_source'],
        dvrl_data['y_source'], 
        batch_size,
        epochs,
        device
    )
    y_hat = pred_func(
        pred_model,
        dvrl_data['x_dev'],
        batch_size,
        device
    )
    baseline_loss = mean_squared_error(dvrl_data['y_dev'], y_hat)

    # Calculate the leave-one-out scores for each sample in the source data
    loo_scores = []
    for i in tqdm(range(len(dvrl_data['x_source']))):
        # Exclude the i-th sample from the source data
        x_train_loo = np.delete(dvrl_data['x_source'], i, axis=0)
        y_train_loo = np.delete(dvrl_data['y_source'], i, axis=0)
        
        # Train the model on the reduced dataset
        if args.input_seq == 'word':
            config = AutoConfig.from_pretrained(args.embedding_model)
            loo_model = MLP(input_feature=config.hidden_size).to(device)
        elif args.input_seq == 'pos':
            loo_model = FeaturesModel().to(device)
        loo_model.load_state_dict(torch.load(f'tmp/init_model{target_prompt_id}.pth'))
        fit_func(
            loo_model,
            x_train_loo,
            y_train_loo,
            batch_size,
            epochs,
            device
        )
        y_hat = pred_func(
            loo_model,
            dvrl_data['x_dev'],
            batch_size,
            device
        )
        loo_loss = mean_squared_error(dvrl_data['y_dev'], y_hat)
        
        # Calculate the difference in MSE loss on the dev set
        loo_score = loo_loss - baseline_loss
        loo_scores.append(loo_score)
    
    # Save the leave-one-out scores
    os.makedirs(f'outputs/Estimated_Data_Values/LOO-{args.input_seq}', exist_ok=True)
    np.save(f'outputs/Estimated_Data_Values/LOO-{args.input_seq}/estimated_data_value{target_prompt_id}.npy', np.array(loo_scores))
    print('Leave-One-Out scores saved.')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="LOO")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_pjname', type=str, default='テスト')
    parser.add_argument('--experiment_name', type=str, default='LOO_DataValueEstimation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--input_seq', type=str, default='pos', choices=['word', 'pos'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)