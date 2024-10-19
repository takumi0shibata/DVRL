"""Training on DVRL"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import json

from dvrl import dvrl_v2
from utils.general_utils import set_seed
from utils.load_data import load_data_DVRL_v2
from dvrl.predictor import MLP


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
    dvrl_data = load_data_DVRL_v2(
        f'data/cross_prompt_attributes/{target_prompt_id}/',
        args.attribute_name,
        args.embedding_model,
        device,
        devsize=args.dev_size
    )

    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    pred_model = MLP(input_feature=dvrl_data['x_source'].shape[1]).to(device)

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'iterations': 1000,
        'activation': nn.Tanh(),
        'layer_number': 5,
        'learning_rate': 1e-3,
        'inner_iterations': 100,
        'batch_size_predictor': 512,
        'wandb': args.wandb,
    }

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{target_prompt_id}_{args.input_seq}',
            config=dict(args._get_kwargs())|dvrl_params
        )

    # Initialize DVRL
    dvrl_class = dvrl_v2.Dvrl(
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
    print('Estimating lambda value...')
    data_value = dvrl_class.dvrl_valuator()

    os.makedirs(f'outputs/Estimated_Lambda/DVRLv2-{args.input_seq}', exist_ok=True)
    np.save(f'outputs/Estimated_Lambda/DVRLv2-{args.input_seq}/estimated_lambda_value{target_prompt_id}.npy', data_value)

    qwk_proposed = dvrl_class.predict(dvrl_data['x_target'], dvrl_data['y_target'])
    qwk_conventional = dvrl_class.predict_baseline(dvrl_data['x_target'], dvrl_data['y_target'])
    # Save results to a file
    results = {
        'QWK Proposed': f'{qwk_proposed:.4f}',
        'QWK Conventional': f'{qwk_conventional:.4f}',
        'Estimated Lambda': data_value.tolist()  # Convert numpy array to list for JSON serialization
    }
    
    # Create a directory for results if it doesn't exist
    os.makedirs('tmp_results', exist_ok=True)
    # Save results to a JSON file
    result_file = f'tmp_results/dvrl_results_prompt{target_prompt_id}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'Results saved to {result_file}')
    
    # Also print results to console
    print(f'QWK Proposed: {qwk_proposed:.4f}')
    print(f'QWK Conventional: {qwk_conventional:.4f}')
    print(f'Estimated Lambda: {data_value}')

    if args.wandb:
        wandb.alert(title=args.wandb_pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-v2-test')
    parser.add_argument('--run_name', type=str, default='DVRL_DataValueEstimation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--input_seq', type=str, default='pos', choices=['word', 'pos'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)