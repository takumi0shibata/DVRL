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

from dvrl import dvrl
from utils.dvrl_utils import calc_qwk, get_dev_sample
from transformers import AutoConfig
from utils.create_embedding_feautres import create_embedding_features
from dvrl.predictor_model import MLP


def main(args):
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
    x_source, y_source = np.concatenate([train_data['essay'], val_data['essay']]), np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    # split test data into dev and test
    x_dev, x_test, y_dev, y_test, dev_ids, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)
    np.save(save_dir + 'dev_ids.npy', dev_ids)

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
    print('X_target: ', x_test.shape)
    print('Y_target: ', y_test.shape)
    print('Y_target max: ', np.max(y_test))
    print('Y_target min: ', np.min(y_test))
    print('================================')


    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    config = AutoConfig.from_pretrained(model_name)
    pred_model = MLP(input_feature=config.hidden_size).to(device)

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
    dvrl_params['batch_size_predictor'] = 256
    dvrl_params['moving_average_window'] = 10
    dvrl_params['moving_average'] = False
    dvrl_params['std_penalty_weight'] = None

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
    y_test_hat = dvrl_class.dvrl_predict(x_test)
    print('Finished data valuation.')


    qwk = calc_qwk(y_test, y_test_hat, test_prompt_id, attribute_name)
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
    parser.add_argument('--experiment_name', type=str, default='DVRL_DomainAdaptation', help='name of the experiment')
    parser.add_argument('--dev_size', type=int, default=30, help='size of the dev set')
    parser.add_argument('--metric', type=str, default='qwk', help='metric to be used for DVRL', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--wandb_pjname', type=str, default='テスト', help='name of the wandb project')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)