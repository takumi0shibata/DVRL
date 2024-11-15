"""
Training on DVRL
問題＆得点ごとにクラスタを分割し，各クラスタに対して，価値を推定する．
"""

import os
import torch
import numpy as np
import argparse
import torch
import wandb
import polars as pl

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.prompt_dataset import PromptDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel

from utils.dvrl_utils import (
    fit_func,
    pred_func,
    calc_qwk,
)


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{target_prompt_id}_{args.input_seq}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(target_prompt_set=args.target_prompt_id, dev_size=args.dev_size, cache_dir='src/.embedding_cache')
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
    # Step2. Training MLP
    ###################################################
    # Select training data
    df = pl.read_csv(f'outputs/dvrl_v3/estimated_values_{target_prompt_id}_{args.input_seq}_{args.seed}.csv')
    dev_qwks = []
    test_qwks = []
    for threshold in np.arange(0.9, -0.1, -0.1):
        # Create predictor
        print('Creating predictor model...')
        if args.input_seq == 'word':
            model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        elif args.input_seq == 'pos':
            model = FeaturesModel().to(device)

        # Filter data
        filtered_clusters = df.filter(pl.col('values') >= threshold)['cluster'].to_numpy()
        if len(filtered_clusters) == 0:
            if args.wandb:
                wandb.log({
                    'Size of training data': 0,
                    'QWK[DEV]': 0.0,
                    'QWK[TEST]': 0.0,
                })
            continue
        cluster_mask = np.isin(train_data['cluster'], filtered_clusters)
        if args.input_seq == 'word':
            selected_train_data = train_data['embedding'][cluster_mask]
        elif args.input_seq == 'pos':
            selected_train_data = train_data['ridley_feature'][cluster_mask]
        selected_train_label = train_data['scaled_score'][cluster_mask]

        # Train model
        print(f'Training model with clusters having values >= {threshold}...')
        fit_func(
            model,
            selected_train_data,
            selected_train_label,
            batch_size=256,
            epochs=100,
            device=device,
        )

        # Predict
        print('Predicting...')
        y_dev_pred = pred_func(
            model,
            dev_data['ridley_feature'] if args.input_seq == 'pos' else dev_data['embedding'],
            batch_size=256,
            device=device,
        )
        y_test_pred = pred_func(
            model,
            test_data['ridley_feature'] if args.input_seq == 'pos' else test_data['embedding'],
            batch_size=256,
            device=device,
        )

        # Calculate QWK
        print('Calculating QWK...')
        dev_qwk = calc_qwk(dev_data['scaled_score'], y_dev_pred, target_prompt_id, args.attribute_name)
        test_qwk = calc_qwk(test_data['scaled_score'], y_test_pred, target_prompt_id, args.attribute_name)

        dev_qwks.append(dev_qwk)
        test_qwks.append(test_qwk)

        print(f'    Dev QWK: {dev_qwk}')
        print(f'    Test QWK: {test_qwk}')

        # log wandb
        data_num = len(selected_train_data)
        if args.wandb:
            wandb.log({
                'Size of training data': data_num,
                'QWK[DEV]': dev_qwk,
                'QWK[TEST]': test_qwk,
            })
    

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='[TEST]DVRL-v3')
    parser.add_argument('--run_name', type=str, default='DVRL_DataValueEstimation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_seq', type=str, default='pos', choices=['word', 'pos'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)