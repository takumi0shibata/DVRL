# coding=utf-8
"""
Training on DVRL class
"""

import os
import platform
import pickle
import torch
import numpy as np
import argparse
import random
import gc
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# import my modules
from configs.configs import Configs
import dvrl.dvrl as dvrl
from utils.general_utils import get_min_max_scores
from utils.my_utils_for_DVRL import load_data, create_data_loader, create_embedding, calc_qwk
from transformers import AutoTokenizer, AutoModel, AutoConfig


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
    # Step1. Create Text Embedding
    ###################################################
    # Load data
    data_path = configs.DATA_PATH2 + str(test_prompt_id) + '/fold-0/'
    print(f'load data from {data_path}...')
    data = load_data(data_path)

    y_train = np.array(data['train']['label'])
    y_dev = np.array(data['dev']['label'])
    y_test = np.array(data['test']['label'])

    minscore, maxscore = get_min_max_scores()[test_prompt_id][attribute_name]
    y_train = (y_train - minscore) / (maxscore - minscore)
    y_dev = (y_dev - minscore) / (maxscore - minscore)
    y_test = (y_test - minscore) / (maxscore - minscore)

    # Create embedding
    os.makedirs(data_path + 'cache/', exist_ok=True)
    pkl_files = [file for file in os.listdir(data_path + 'cache/') if file.endswith('.pkl')]
    model_name = 'microsoft/deberta-v3-large'
    config = AutoConfig.from_pretrained(model_name)
    if len(pkl_files) == 0:
        # Load embedding model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        train_loader = create_data_loader(data['train']['feature'], tokenizer, max_length=512, batch_size=32)
        dev_loader = create_data_loader(data['dev']['feature'], tokenizer, max_length=512, batch_size=32)
        test_loader = create_data_loader(data['test']['feature'], tokenizer, max_length=512, batch_size=32)

        # Create embedding
        print('[Train]')
        train_features = create_embedding(train_loader, model, device)
        print('[Dev]')
        dev_features = create_embedding(dev_loader, model, device)
        print('[Test]')
        test_features = create_embedding(test_loader, model, device)
        
        torch.cuda.empty_cache()
        gc.collect()

        # Save embedding
        with open(data_path + 'cache/train_features.pkl', 'wb') as f:
            pickle.dump(train_features, f)
        with open(data_path + 'cache/dev_features.pkl', 'wb') as f:
            pickle.dump(dev_features, f)
        with open(data_path + 'cache/test_features.pkl', 'wb') as f:
            pickle.dump(test_features, f)
    else:
        print('Loading embedding from cache...')
        train_features = pickle.load(open(data_path + 'cache/train_features.pkl', 'rb'))
        dev_features = pickle.load(open(data_path + 'cache/dev_features.pkl', 'rb'))
        test_features = pickle.load(open(data_path + 'cache/test_features.pkl', 'rb'))

    ###################################################
    # Step1 extra. 後で消す↓
    ###################################################
    # Load data
    data_path = configs.DATA_PATH2 + str(2) + '/fold-0/'
    print(f'load data from {data_path}...')
    data2 = load_data(data_path)

    y_train2 = np.array(data2['train']['label'])
    y_dev2 = np.array(data2['dev']['label'])
    y_test2 = np.array(data2['test']['label'])

    minscore, maxscore = get_min_max_scores()[2][attribute_name]
    y_train2 = (y_train2 - minscore) / (maxscore - minscore)
    y_dev2 = (y_dev2 - minscore) / (maxscore - minscore)
    y_test2 = (y_test2 - minscore) / (maxscore - minscore)

    # Create embedding
    os.makedirs(data_path + 'cache/', exist_ok=True)
    pkl_files = [file for file in os.listdir(data_path + 'cache/') if file.endswith('.pkl')]
    model_name = 'microsoft/deberta-v3-large'
    config = AutoConfig.from_pretrained(model_name)

    print('Loading embedding from cache...')
    train_features2 = pickle.load(open(data_path + 'cache/train_features.pkl', 'rb'))
    dev_features2 = pickle.load(open(data_path + 'cache/dev_features.pkl', 'rb'))
    test_features2 = pickle.load(open(data_path + 'cache/test_features.pkl', 'rb'))

    # プロンプト１とプロンプト２のデータを結合
    train_features = np.concatenate([train_features, train_features2], axis=0)
    y_train = np.concatenate([y_train, y_train2[::-1]], axis=0)
    ###################################################
    # Step1 extra. 後で消す↑
    ###################################################

    # print info
    print('================================')
    print('X_train: ', train_features.shape)
    print('Y_train: ', y_train.shape)
    print('Y_train max: ', np.max(y_train))
    print('Y_train min: ', np.min(y_train))

    print('================================')
    print('X_dev: ', dev_features.shape)
    print('Y_dev: ', y_dev.shape)
    print('Y_dev max: ', np.max(y_dev))
    print('Y_dev min: ', np.min(y_dev))

    print('================================')
    print('X_test: ', test_features.shape)
    print('Y_test: ', y_test.shape)
    print('Y_test max: ', np.max(y_test))
    print('Y_test min: ', np.min(y_test))
    print('================================')


    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    pred_model = nn.Sequential(
        nn.Linear(config.hidden_size, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    ).to(device)

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
    dvrl_class = dvrl.Dvrl(train_features, y_train, dev_features, y_dev, pred_model, dvrl_params, device, test_prompt_id)

    # Train DVRL
    print('Training DVRL...')
    rewards_history, losses_history = dvrl_class.train_dvrl('qwk')

    # Estimate data value
    print('Estimating data value...')
    data_value = dvrl_class.dvrl_valuator(train_features, y_train)
    np.save('./tmp/estimated_data_value.npy', data_value)

    # Pridicts with DVRl
    y_test_hat = dvrl_class.dvrl_predict(test_features)
    print('Finished data valuation.')


    qwk = calc_qwk(y_test, y_test_hat, test_prompt_id, attribute_name)
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