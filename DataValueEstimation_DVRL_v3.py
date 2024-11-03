"""
Training on DVRL
問題＆得点ごとにクラスタを分割し，各クラスタに対して，価値を推定する．
"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import polars as pl

from dvrl import dvrl_v3
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.prompt_dataset import PromptDataset
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
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(target_prompt_set=args.target_prompt_id, dev_size=args.dev_size)
    if args.input_seq == 'pos':
        train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
        dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
        test_data['ridley_feature'] = np.concatenate([test_data['feature'], test_data['readability']], axis=1)
    print(f'    Number of training samples: {len(train_data['essay_id'])}')
    print(f'    Number of dev samples: {len(dev_data['essay_id'])}')
    print(f'    Number of test samples: {len(test_data['essay_id'])}')

    # Load prompt data
    print('Loading prompt data...')
    prompt_dataset = PromptDataset('data/prompts.csv')
    prompt_data = prompt_dataset.load()
    print(f'    Number of clusters: {len(np.unique(prompt_data["cluster"]))}')

    # Assign cluster
    train_cluster = []
    for i in range(len(train_data['essay_id'])):
        essay_set = train_data['essay_set'][i]
        score = train_data['original_score'][i]

        is_match_essay_set = (prompt_data['essay_set'] == essay_set)
        is_match_score = (prompt_data['score'] == score)
        cluster = prompt_data['cluster'][is_match_essay_set & is_match_score]
        train_cluster.append(cluster[0])
    train_data['cluster'] = np.array(train_cluster)
    print(f'    Number of training clusters: {len(np.unique(train_data["cluster"]))}')

    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    if args.input_seq == 'word':
        pred_model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
    elif args.input_seq == 'pos':
        pred_model = FeaturesModel().to(device)

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'comb_dim': 64,
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

    
    if args.input_seq == 'word':
        x_train = train_data['embedding']
        x_dev = dev_data['embedding']
    elif args.input_seq == 'pos':
        x_train = train_data['ridley_feature']
        x_dev = dev_data['ridley_feature']
    
    # Initialize DVRL
    dvrl_class = dvrl_v3.Dvrl(
        train_data,
        x_train,
        train_data['scaled_score'],
        x_dev,
        dev_data['scaled_score'],
        prompt_data,
        pred_model,
        dvrl_params,
        device,
        target_prompt_id
    )

    # Train DVRL
    print('Training DVRL...')
    outputs = dvrl_class.train_dvrl(args.metric)

    # Save DVRL
    print('Saving DVRL...')
    output_dir = './outputs/dvrl_v3'
    os.makedirs(output_dir, exist_ok=True)
    pl.DataFrame(outputs).write_csv(os.path.join(output_dir, f'estimated_values_{target_prompt_id}_{args.input_seq}.csv'))

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
    parser.add_argument('--input_seq', type=str, default='word', choices=['word', 'pos'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)