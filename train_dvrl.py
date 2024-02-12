# coding=utf-8
"""
Training on DVRL class
"""

import os
import platform
import torch
import numpy as np
import argparse
import random
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# import my modules
from configs.configs import Configs
import dvrl.dvrl as dvrl
from utils.dvrl_utils import calc_qwk
from transformers import AutoConfig
from utils.create_embedding_feautres import create_embedding_features
from models.predictor_model import EssayScorer


def main():
    ###################################################
    # Step0. Set UP
    ###################################################
    warnings.filterwarnings('ignore')
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory')
    parser.add_argument('--experiment_name', type=str, default='DVRL', help='name of the experiment')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed

    # Load configs
    configs = Configs()

    # Set device
    pf = platform.system()
    if pf == 'Windows' or pf == 'Linux':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif pf == 'Darwin':
        device = torch.device('mps')
    else:
        raise Exception('Unknown platform')
    
    #fix random seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # print info
    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Attribute: {}".format(attribute_name))
    print("Seed: {}".format(seed))
    print("Device: {}".format(device))


    ###################################################
    # Step1. Create/Load Text Embedding
    ###################################################
    # Load data
    noise_prompt = 2
    data_path1 = configs.DATA_PATH2 + str(test_prompt_id) + '/fold-0/'
    data_path2 = configs.DATA_PATH2 + str(noise_prompt) + '/fold-0/'
    model_name = 'microsoft/deberta-v3-large'

    train_features1, dev_features1, test_features1, y_train1, y_dev1, y_test1 = create_embedding_features(data_path1, test_prompt_id, attribute_name, model_name, device)
    train_features2, dev_features2, test_features2, y_train2, y_dev2, y_test2 = create_embedding_features(data_path2, noise_prompt, attribute_name, model_name, device)

    # Concatenate data of prompt 1 and prompt 2
    train_features = np.concatenate([train_features1, train_features2], axis=0)
    y_train = np.concatenate([y_train1, y_train2[::-1]], axis=0)

    # print info
    print('================================')
    print('X_train: ', train_features.shape)
    print('Y_train: ', y_train.shape)
    print('Y_train max: ', np.max(y_train))
    print('Y_train min: ', np.min(y_train))

    print('================================')
    print('X_dev: ', dev_features1.shape)
    print('Y_dev: ', y_dev1.shape)
    print('Y_dev max: ', np.max(y_dev1))
    print('Y_dev min: ', np.min(y_dev1))

    print('================================')
    print('X_test: ', test_features1.shape)
    print('Y_test: ', y_test1.shape)
    print('Y_test max: ', np.max(y_test1))
    print('Y_test min: ', np.min(y_test1))
    print('================================')


    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    config = AutoConfig.from_pretrained(model_name)
    pred_model = EssayScorer(input_feature=config.hidden_size).to(device)

    # Network parameters
    print('Initialize DVRL class')
    dvrl_params = {}
    dvrl_params['hidden_dim'] = 100
    dvrl_params['comb_dim'] = 10
    dvrl_params['iterations'] = 30
    dvrl_params['activation'] = nn.ReLU()
    dvrl_params['layer_number'] = 5
    dvrl_params['learning_rate'] = 0.01
    dvrl_params['batch_size'] = 2000
    dvrl_params['inner_iterations'] = 100
    dvrl_params['batch_size_predictor'] = 256


    # Initialize DVRL
    dvrl_class = dvrl.Dvrl(train_features, y_train, dev_features1, y_dev1, pred_model, dvrl_params, device, test_prompt_id)

    # Train DVRL
    print('Training DVRL...')
    rewards_history, losses_history = dvrl_class.train_dvrl('qwk')

    # Estimate data value
    print('Estimating data value...')
    data_value = dvrl_class.dvrl_valuator(train_features, y_train)
    np.save('./tmp/estimated_data_value.npy', data_value)

    # Pridicts with DVRl
    y_test_hat = dvrl_class.dvrl_predict(test_features1)
    print('Finished data valuation.')


    qwk = calc_qwk(y_test1, y_test_hat, test_prompt_id, attribute_name)
    print(f'QWK: {qwk: .4f}')
    print(f'Data Value: {data_value}')
    
    # plot loss
    epochs = list(range(1, dvrl_params['iterations']+1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rewards_history, label='Train', color='red')
    plt.title('Rewards over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./tmp/rewards_history.png')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses_history, label='Train', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./tmp/losses_history.png')


if __name__ == '__main__':
    main()