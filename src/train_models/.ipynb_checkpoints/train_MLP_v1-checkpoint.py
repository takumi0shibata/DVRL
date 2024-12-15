'''
This script trains MLP model.
    - for DeBERTa-v3-large embedding vectors
    - for mannualy designed features
'''

import os
import torch
import numpy as np
import argparse
import torch
import wandb
import polars as pl
from sklearn.metrics import mean_squared_error

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
    remove_top_p_sample
)


def train_and_evaluate(
    train_data,
    dev_data,
    test_data,
    target_prompt_id,
    weights,
    batch_size,
    epochs,
    device,
    attribute_name,
    args,
):
    
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    # Create predictor
    print('Creating predictor model...')
    mlp_data = {}
    mlp_data['y_source'] = train_data['scaled_score']
    mlp_data['y_dev'] = dev_data['scaled_score']
    mlp_data['y_test'] = test_data['scaled_score']
    if args.pred_model == 'mlp':
        pred_model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        mlp_data['x_source'] = train_data['embedding']
        mlp_data['x_dev'] = dev_data['embedding']
        mlp_data['x_test'] = test_data['embedding']
    elif args.pred_model == 'features_model':
        pred_model = FeaturesModel().to(device)
        train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
        dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
        test_data['ridley_feature'] = np.concatenate([test_data['feature'], test_data['readability']], axis=1)
        mlp_data['x_source'] = train_data['ridley_feature']
        mlp_data['x_dev'] = dev_data['ridley_feature']
        mlp_data['x_test'] = test_data['ridley_feature']

    fit_func(
        pred_model,
        mlp_data['x_source'][weights],
        mlp_data['y_source'][weights],
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # For dev
    y_pred_dev = pred_func(
        pred_model,
        mlp_data['x_dev'],
        batch_size=batch_size,
        device=device
    )
    # For test
    y_pred_test = pred_func(
        pred_model,
        mlp_data['x_test'],
        batch_size=batch_size,
        device=device
    )
    test_qwk = calc_qwk(mlp_data['y_test'], y_pred_test, target_prompt_id, attribute_name)
    dev_mse = mean_squared_error(mlp_data['y_dev'], y_pred_dev)
    return test_qwk, dev_mse

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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
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
    estimated_data_value = np.load(f'outputs/dvrl_v1/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}.npy')
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ##################################################
        # データの価値が低いものを削除
        ##################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=False,
        )
        qwk_high, dev_loss_high = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            target_prompt_id,
            weights,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args,
        )

        ##################################################
        # データの価値が高いものを削除
        ##################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=True,
        )
        qwk_low, dev_loss_low = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            target_prompt_id,
            weights,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args,
        )

        print(f'p: {p_val:.1f}, QWK[High]: {qwk_high:.3f}, QWK[Low]: {qwk_low:.3f}, Dev Loss[High]: {dev_loss_high:.10f}, Dev Loss[Low]: {dev_loss_low:.10f}')

        if args.wandb:
            wandb.log({
                'p': p_val,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
            })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='Training')
    parser.add_argument('--run_name', type=str, default='train-MLP')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pred_model',type=str, default='mlp', choices=['mlp', 'features_model'])
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)