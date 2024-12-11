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

import umap
from sklearn.cluster import KMeans

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
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
            name=args.run_name + f'_{target_prompt_id}_{args.pred_model}_seed{args.seed}_cluster{args.n_clusters}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        dev_size=args.dev_size,
        cache_dir='src/.embedding_cache',
        embedding_model=args.embedding_model,
        add_pos=False,
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')


    # Assign cluster based on embedding similarity
    print('Assigning clusters based on embedding similarity...')
    embeddings = train_data['embedding']

    # UMAPで5次元に次元削減
    reducer = umap.UMAP(n_components=5, random_state=args.seed)
    reduced_embeddings = reducer.fit_transform(embeddings)
    # 埋め込みベクトルを正規化
    norm_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
    # 得点データを結合
    norm_embeddings = np.concatenate((norm_embeddings, train_data['scaled_score'].reshape(-1, 1)), axis=1)
    # クラスタリング
    clustering = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(reduced_embeddings)  # n_clusters should be tuned

    train_data['cluster'] = clustering.labels_

    print(f'    Number of training clusters: {len(np.unique(train_data["cluster"]))}')

    ###################################################
    # Step2. Training MLP
    ###################################################
    # Select training data
    df = pl.read_csv(f'outputs/dvrl_v4/estimated_values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_cluster{args.n_clusters}.csv')
    for threshold in np.arange(0.0, 1.1, 0.1):
        ############################################
        # Remove high values
        ############################################
        # Create predictor
        print('Creating predictor model...')
        if args.pred_model == 'mlp':
            model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        elif args.pred_model == 'features_model':
            model = FeaturesModel().to(device)

        # Filter data
        threshold_percentile = df.quantile(1 - threshold, 'nearest')['values'].item() # 昇順にソートされている
        filtered_clusters = df.filter(pl.col('values') <= threshold_percentile)['cluster'].to_numpy()
        cluster_mask = np.isin(train_data['cluster'], filtered_clusters)
        if args.pred_model == 'mlp':
            selected_train_data = train_data['embedding'][cluster_mask]
        elif args.pred_model == 'features_model':
            train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
            dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
            test_data['ridley_feature'] = np.concatenate([test_data['feature'], test_data['readability']], axis=1)
            selected_train_data = train_data['ridley_feature'][cluster_mask]
        selected_train_label = train_data['scaled_score'][cluster_mask]

        # Train model
        print(f'Training model with threshold={threshold_percentile}...')
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
            dev_data['ridley_feature'] if args.pred_model == 'features_model' else dev_data['embedding'],
            batch_size=256,
            device=device,
        )
        y_test_pred = pred_func(
            model,
            test_data['ridley_feature'] if args.pred_model == 'features_model' else test_data['embedding'],
            batch_size=256,
            device=device,
        )

        # Calculate QWK
        print('Calculating QWK...')
        dev_qwk = calc_qwk(dev_data['scaled_score'], y_dev_pred, target_prompt_id, args.attribute_name)
        test_qwk = calc_qwk(test_data['scaled_score'], y_test_pred, target_prompt_id, args.attribute_name)

        print(f'    Dev QWK: {dev_qwk}')
        print(f'    Test QWK: {test_qwk}')

        # log wandb
        if args.wandb:
            wandb.log({
                'LOW QWK[DEV]': dev_qwk,
                'LOW QWK[TEST]': test_qwk,
            })
        
        ############################################
        # Remove low values
        ############################################
        # Create predictor
        print('Creating predictor model...')
        if args.pred_model == 'mlp':
            model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        elif args.pred_model == 'features_model':
            model = FeaturesModel().to(device)
        
        # Filter data
        threshold_percentile = df.quantile(threshold, 'nearest')['values'].item()
        filtered_clusters = df.filter(pl.col('values') >= threshold_percentile)['cluster'].to_numpy()
        cluster_mask = np.isin(train_data['cluster'], filtered_clusters)
        if args.pred_model == 'mlp':
            selected_train_data = train_data['embedding'][cluster_mask]
        elif args.pred_model == 'features_model':
            selected_train_data = train_data['ridley_feature'][cluster_mask]
        selected_train_label = train_data['scaled_score'][cluster_mask]

        # Train model
        print(f'Training model with threshold={threshold_percentile}...')
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
            dev_data['ridley_feature'] if args.pred_model == 'features_model' else dev_data['embedding'],
            batch_size=256,
            device=device,
        )
        y_test_pred = pred_func(
            model,
            test_data['ridley_feature'] if args.pred_model == 'features_model' else test_data['embedding'],
            batch_size=256,
            device=device,
        )

        # Calculate QWK
        print('Calculating QWK...')
        dev_qwk = calc_qwk(dev_data['scaled_score'], y_dev_pred, target_prompt_id, args.attribute_name)
        test_qwk = calc_qwk(test_data['scaled_score'], y_test_pred, target_prompt_id, args.attribute_name)

        print(f'    Dev QWK: {dev_qwk}')
        print(f'    Test QWK: {test_qwk}')

        # log wandb
        if args.wandb:
            wandb.log({
                'HIGH QWK[DEV]': dev_qwk,
                'HIGH QWK[TEST]': test_qwk,
            })

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V4')
    parser.add_argument('--run_name', type=str, default='Training')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_clusters', type=int, default=1000)
    parser.add_argument('--pred_model',type=str, default='mlp', choices=['mlp', 'features_model'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)