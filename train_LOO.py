"""Training on LOO"""

import os
import torch
import numpy as np
import argparse
import torch
import wandb
from tqdm import tqdm

from utils.dvrl_utils import get_dev_sample, fit_func, pred_func
from transformers import AutoConfig
from utils.create_embedding_feautres import create_embedding_features
from utils.general_utils import set_seed
from dvrl.predictor_model import MLP
from sklearn.metrics import mean_squared_error


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    save_dir = args.save_dir + '/'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)
    set_seed(seed)

    ###################################################
    # Step1. Create/Load Text Embedding
    ###################################################
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    x_source = np.concatenate([train_data['essay'], val_data['essay']])
    y_source = np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    # split test data into dev and test
    x_dev, _, y_dev, _, _, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

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

    ###################################################
    # Step2. Training LOO
    ###################################################
    wandb.init(project=args.pjname, name=args.run_name+str(test_prompt_id), config=args)
    # Create predictor
    print('Creating predictor model...')
    config = AutoConfig.from_pretrained(model_name)
    pred_model = MLP(input_feature=config.hidden_size).to(device)
    torch.save(pred_model.state_dict(), f'tmp/init_model{test_prompt_id}.pth')

    # Calculate the leave-one-out scores for source data
    fit_func(pred_model, x_source, y_source, batch_size, epochs, device)
    y_hat = pred_func(pred_model, x_dev, batch_size, device)
    baseline_loss = mean_squared_error(y_dev, y_hat)

    # Calculate the leave-one-out scores for each sample in the source data
    loo_scores = []
    for i in tqdm(range(len(x_source))):
        # Exclude the i-th sample from the source data
        x_train_loo = np.delete(x_source, i, axis=0)
        y_train_loo = np.delete(y_source, i, axis=0)
        
        # Train the model on the reduced dataset
        loo_model = MLP(input_feature=config.hidden_size).to(device)
        loo_model.load_state_dict(torch.load(f'tmp/init_model{test_prompt_id}.pth'))
        fit_func(loo_model, x_train_loo, y_train_loo, batch_size, epochs, device)
        y_hat = pred_func(loo_model, x_dev, batch_size, device)
        loo_loss = mean_squared_error(y_dev, y_hat)
        
        # Calculate the difference in MSE loss on the dev set
        loo_score = loo_loss - baseline_loss
        loo_scores.append(loo_score)
    
    # Save the leave-one-out scores
    np.save(save_dir + f'estimated_data_value{test_prompt_id}.npy', np.array(loo_scores))
    print('Leave-One-Out scores saved.')
    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="LOO")
    parser.add_argument('--pjname', type=str, default='LOO', help='name of the wandb project')
    parser.add_argument('--run_name', type=str, default='LOO_MLP', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--save_dir', type=str, default='outputs/Estimated_Data_Values/LOO-MLP', help='data value directory')
    parser.add_argument('--dev_size', type=int, default=30, help='size of the dev set')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--device', type=str, default='cuda', help='device to be used', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)