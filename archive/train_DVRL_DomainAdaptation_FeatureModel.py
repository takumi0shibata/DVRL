"""Training on DVRL class"""

import os
import platform
import torch
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import warnings

from dvrl import dvrl
from utils.dvrl_utils import calc_qwk, get_dev_sample
from transformers import AutoConfig
from utils.create_embedding_feautres import create_embedding_features, normalize_scores
from utils.read_data import get_readability_features, get_linguistic_features, get_features_by_id, scale_features
from dvrl.predictor_model import MLP
from models.transfomer_enc import FeatureModel


def main(args):
    warnings.filterwarnings("ignore")
    ###################################################
    # Step0. Set UP
    ###################################################
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    save_dir = args.output_dir + args.experiment_name + '/'
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    pf = platform.system()
    if pf == 'Windows' or pf == 'Linux':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif pf == 'Darwin':
        device = torch.device('mps')
    else:
        raise Exception('Unknown platform')
    
    # fix random seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ###################################################
    # Step1. Create/Load Text Embedding
    ###################################################
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    # split test data into dev and test
    _, _, _, _, dev_idx, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

    source_ids = np.concatenate([train_data['essay_id'], val_data['essay_id']])
    dev_ids = test_data['essay_id'][dev_idx]
    target_ids = np.setdiff1d(test_data['essay_id'], dev_ids)

    # load readability and linguistic features
    readability_features = get_readability_features('data/allreadability.pickle')
    linguistic_features = get_linguistic_features('data/hand_crafted_v3.csv')

    x_source_readability = get_features_by_id(readability_features, source_ids, 'dim1').drop('dim1', axis=1).to_numpy()
    x_dev_readability = get_features_by_id(readability_features, dev_ids, 'dim1').drop('dim1', axis=1).to_numpy()
    x_target_readability = get_features_by_id(readability_features, target_ids, 'dim1').drop('dim1', axis=1).to_numpy()

    x_source_linguistic = get_features_by_id(linguistic_features, source_ids, 'item_id')
    x_dev_linguistic = get_features_by_id(linguistic_features, dev_ids, 'item_id')
    x_target_linguistic = get_features_by_id(linguistic_features, target_ids, 'item_id')

    y_source = x_source_linguistic['score'].to_numpy()
    y_source_prompt = x_source_linguistic['prompt_id'].to_numpy()
    y_source = normalize_scores(y_source, y_source_prompt, 'score')

    y_dev = x_dev_linguistic['score'].to_numpy()
    y_dev_prompt = x_dev_linguistic['prompt_id'].to_numpy()
    y_dev = normalize_scores(y_dev, y_dev_prompt, 'score')

    y_target = x_target_linguistic['score'].to_numpy()
    y_target_prompt = x_target_linguistic['prompt_id'].to_numpy()
    y_target = normalize_scores(y_target, y_target_prompt, 'score')

    x_source_linguistic_scaled = scale_features(x_source_linguistic).drop(['item_id', 'prompt_id', 'score'], axis=1).to_numpy()
    x_dev_linguistic_scaled = scale_features(x_dev_linguistic).drop(['item_id', 'prompt_id', 'score'], axis=1).to_numpy()
    x_target_linguistic_scaled = scale_features(x_target_linguistic).drop(['item_id', 'prompt_id', 'score'], axis=1).to_numpy()

    x_source = np.concatenate([x_source_readability, x_source_linguistic_scaled], axis=1)
    x_dev = np.concatenate([x_dev_readability, x_dev_linguistic_scaled], axis=1)
    x_target = np.concatenate([x_target_readability, x_target_linguistic_scaled], axis=1)

    

    # print info
    print('================================')
    print('X_source: ', x_source.shape)
    print('Y_source: ', y_source.shape)
    print('Y_source max: ', np.max(y_source))
    print('Y_source min: ', np.min(y_source))

    print('================================')
    print('X_dev: ', x_dev.shape)
    print('Y_dev: ', y_dev.shape)
    print('Y_dev max: ', np.max(y_dev))
    print('Y_dev min: ', np.min(y_dev))

    print('================================')
    print('X_target: ', x_target.shape)
    print('Y_target: ', y_target.shape)
    print('Y_target max: ', np.max(y_target))
    print('Y_target min: ', np.min(y_target))
    print('================================')


    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    pred_model = FeatureModel(
        readability_size=x_source_readability.shape[1],
        linguistic_size=x_source_linguistic_scaled.shape[1]
        ).to(device)

    # Network parameters
    print('Initialize DVRL class')
    dvrl_params = {}
    dvrl_params['hidden_dim'] = 100
    dvrl_params['comb_dim'] = 10
    dvrl_params['iterations'] = 1000
    dvrl_params['activation'] = nn.Tanh()
    dvrl_params['layer_number'] = 5
    dvrl_params['learning_rate'] = 0.001
    dvrl_params['batch_size'] = 10000
    dvrl_params['inner_iterations'] = 100
    dvrl_params['batch_size_predictor'] = 512
    dvrl_params['moving_average_window'] = 10
    dvrl_params['moving_average'] = False
    dvrl_params['std_penalty_weight'] = None
    dvrl_params['test_prompt_id'] = test_prompt_id
    dvrl_params['seed'] = seed
    dvrl_params['dev_size'] = args.dev_size

    # Init wandb
    wandb.init(project=args.wandb_pjname, name=args.experiment_name, config=dvrl_params)

    # Initialize DVRL
    dvrl_class = dvrl.Dvrl(x_source, y_source, x_dev, y_dev, pred_model, dvrl_params, device, test_prompt_id)

    # Train DVRL
    print('Training DVRL...')
    dvrl_class.train_dvrl(args.metric)

    # Estimate data value
    print('Estimating data value...')
    data_value = dvrl_class.dvrl_valuator(x_source, y_source)
    np.save(save_dir + 'estimated_data_value.npy', data_value)

    # Pridicts with DVRl
    y_test_hat = dvrl_class.dvrl_predict(x_target)
    print('Finished data valuation.')


    qwk = calc_qwk(y_target, y_test_hat, test_prompt_id, attribute_name)
    print(f'QWK: {qwk: .4f}')
    print(f'Data Value: {data_value}')

    wandb.alert(title=args.wandb_pjname, text='Training finished!')
    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory')
    parser.add_argument('--experiment_name', type=str, default='DVRL_DomainAdaptation_FeatureModel', help='name of the experiment')
    parser.add_argument('--dev_size', type=int, default=30, help='size of the dev set')
    parser.add_argument('--metric', type=str, default='qwk', help='metric to be used for DVRL', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--wandb_pjname', type=str, default='DVRL-FeatureModel', help='name of the wandb project')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)