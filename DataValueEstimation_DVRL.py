"""Training on DVRL"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb

from dvrl import dvrl
from transformers import AutoConfig
from utils.general_utils import set_seed
from utils.load_data import load_data_DVRL, load_data_PAES
from dvrl.predictor import MLP
from models.features import FeaturesModel


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    ###################################################
    # Step1. Load Data
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
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    if args.input_seq == 'word':
        config = AutoConfig.from_pretrained(args.embedding_model)
        pred_model = MLP(input_feature=config.hidden_size).to(device)
    elif args.input_seq == 'pos':
        pred_model = FeaturesModel().to(device)

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'comb_dim': 10,
        'iterations': 1000,
        'activation': nn.Tanh(),
        'layer_number': 5,
        'learning_rate': 0.001,
        'batch_size': 10000,
        'inner_iterations': 100,
        'batch_size_predictor': 512,
        # 'moving_average_window': 10,
        'wandb': args.wandb,
    }

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{target_prompt_id}_{args.input_seq}',
            config=dict(args._get_kwargs())|dvrl_params
        )

    # Initialize DVRL
    dvrl_class = dvrl.Dvrl(
        dvrl_data,
        pred_model,
        dvrl_params,
        device,
        target_prompt_id
    )

    # Train DVRL
    print('Training DVRL...')
    dvrl_class.train_dvrl(args.metric)

    # Estimate data value
    print('Estimating data value...')
    data_value = dvrl_class.dvrl_valuator(dvrl_data['x_source'], dvrl_data['y_source'])

    os.makedirs(f'outputs/Estimated_Data_Values/DVRL-{args.input_seq}', exist_ok=True)
    np.save(f'outputs/Estimated_Data_Values/DVRL-{args.input_seq}/estimated_data_value{target_prompt_id}.npy', data_value)

    if args.wandb:
        wandb.alert(title=args.wandb_pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='テスト')
    parser.add_argument('--run_name', type=str, default='DVRL_DataValueEstimation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--input_seq', type=str, default='pos', choices=['word', 'pos'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)