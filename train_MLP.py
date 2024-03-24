"""Training on DVRL class"""

import os
import platform
import torch
import numpy as np
import argparse
import random
import torch
import wandb

from utils.dvrl_utils import calc_qwk, get_dev_sample, fit_func, pred_func
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

    # ###################################################
    # # Step1. Create/Load Text Embedding # This is dev only
    # ###################################################
    # # Load data
    # data_path = args.data_dir + str(test_prompt_id) + '/'
    # model_name = args.embedding_model

    # _, _, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    # # split test data into dev and test
    # x_dev, x_test, y_dev, y_test, _, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)


    # print('================================')
    # print('X_dev: ', x_dev.shape)
    # print('Y_dev: ', y_dev.shape)
    # print('Y_dev max: ', np.max(y_dev))
    # print('Y_dev min: ', np.min(y_dev))

    # print('================================')
    # print('X_target: ', x_test.shape)
    # print('Y_target: ', y_test.shape)
    # print('Y_target max: ', np.max(y_test))
    # print('Y_target min: ', np.min(y_test))
    # print('================================')

    ###################################################
    # Step1. Create/Load Text Embedding # This is full data
    ###################################################
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    x_source, y_source = np.concatenate([train_data['essay'], val_data['essay']]), np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    # split test data into dev and test
    x_dev, x_test, y_dev, y_test, _, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

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
    model = MLP(input_feature=config.hidden_size).to(device)

    # Init wandb
    wandb.init(project=args.wandb_pjname, name=f'MLP-fullsource{args.dev_size}-{test_prompt_id}', config=dict(args._get_kwargs()))

    # Train
    best_qwk = -1
    best_dev_qwk = -1
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        history = fit_func(model, x_source, y_source, batch_size=256, epochs=1, device=device)

        y_pred_dev = pred_func(model, x_dev, batch_size=256, device=device)
        y_pred_test = pred_func(model, x_test, batch_size=256, device=device)
        dev_qwk = calc_qwk(y_dev, y_pred_dev, test_prompt_id, 'score')
        test_qwk = calc_qwk(y_test, y_pred_test, test_prompt_id, 'score')

        if dev_qwk > best_dev_qwk:
            best_dev_qwk = dev_qwk
            best_qwk = test_qwk

        wandb.log({'QWK[DEV]': dev_qwk, 'QWK[TEST]': test_qwk, 'QWK[BEST TEST]': best_qwk, 'train_loss': history[0]})


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
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--wandb_pjname', type=str, default='MLP-fullsource', help='name of the wandb project')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)