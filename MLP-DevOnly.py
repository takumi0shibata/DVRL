"""Training on DVRL class"""

import torch
import numpy as np
import argparse
import torch
from transformers import AutoConfig
import wandb

from utils.dvrl_utils import calc_qwk, get_dev_sample, fit_func, pred_func
from utils.general_utils import set_seed
from utils.create_embedding_feautres import create_embedding_features
from dvrl.predictor_model import MLP


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    batch_size = args.batch_size
    seed = args.seed
    set_seed(seed)
    device = torch.device(args.device)

    ###################################################
    # Step1. Create/Load Text Embedding # This is dev only
    ###################################################
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    _, _, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    # split test data into dev and test
    x_dev, x_test, y_dev, y_test, _, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)


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
    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)

    # Train
    best_qwk = -1
    best_dev_qwk = -1
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        history = fit_func(model, x_dev, y_dev, batch_size=batch_size, epochs=1, device=device)

        y_pred_dev = pred_func(model, x_dev, batch_size=batch_size, device=device)
        y_pred_test = pred_func(model, x_test, batch_size=batch_size, device=device)
        dev_qwk = calc_qwk(y_dev, y_pred_dev, test_prompt_id, attribute_name)
        test_qwk = calc_qwk(y_test, y_pred_test, test_prompt_id, attribute_name)

        if dev_qwk > best_dev_qwk:
            best_dev_qwk = dev_qwk
            best_qwk = test_qwk

        wandb.log({'QWK[DEV]': dev_qwk, 'QWK[TEST]': test_qwk, 'QWK[BEST TEST]': best_qwk, 'train_loss': history[0]})

    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='MLP-DevOnly', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)