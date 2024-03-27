"""This script trains the PAES_on_torch model on the given prompt and attribute."""

import os
import platform
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

# import my modules
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences, set_seed
from models.paes import PAES, tinyPAES
from utils.evaluation import train_model, evaluate_model

def main(args):
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    device = torch.device(args.device)
    set_seed(seed)

    data_path = args.data_dir
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    epochs = args.epochs
    batch_size = args.batch_size

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': args.features_path,
        'readability_path': args.readability_path
    }

    ########################################################
    from utils.create_embedding_feautres import create_embedding_features
    from utils.dvrl_utils import get_dev_sample
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    _, _, test_data = create_embedding_features(data_path, attribute_name, args.embedding_model, device)
    _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)
    ########################################################

    # Read data
    pos_vocab = read_pos_vocab(read_configs)
    train_data, dev_data, test_data = read_essays_single_score(read_configs, pos_vocab, attribute_name)


    # Get max sentence length and max sentence number
    max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])

    # Scale down the scores
    train_data['y_scaled'] = get_single_scaled_down_score(train_data['data_y'], train_data['prompt_ids'], attribute_name)
    dev_data['y_scaled'] = get_single_scaled_down_score(dev_data['data_y'], dev_data['prompt_ids'], attribute_name)
    test_data['y_scaled'] = get_single_scaled_down_score(test_data['data_y'], test_data['prompt_ids'], attribute_name)

    # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
    X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

    X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))

    # convert to tensor
    X_source = torch.tensor(np.concatenate([X_train_pos, X_dev_pos, X_test_pos[dev_idx]], axis=0), dtype=torch.long)
    X_dev = torch.tensor(X_test_pos[dev_idx], dtype=torch.long)
    X_target = torch.tensor(X_test_pos[test_idx], dtype=torch.long)

    X_source_linguistic_features = torch.tensor(np.concatenate([train_data['features_x'], dev_data['features_x'], np.array(test_data['features_x'])[dev_idx]], axis=0), dtype=torch.float)
    X_dev_linguistic_features = torch.tensor(np.array(test_data['features_x'])[dev_idx], dtype=torch.float)
    X_target_linguistic_features = torch.tensor(np.array(test_data['features_x'])[test_idx], dtype=torch.float)

    X_source_readability = torch.tensor(np.concatenate([train_data['readability_x'], dev_data['readability_x'], np.array(test_data['readability_x'])[dev_idx]], axis=0), dtype=torch.float)
    X_dev_readability = torch.tensor(np.array(test_data['readability_x'])[dev_idx], dtype=torch.float)
    X_target_readability = torch.tensor(np.array(test_data['readability_x'])[test_idx], dtype=torch.float)

    Y_train = torch.tensor(np.concatenate([train_data['y_scaled'], dev_data['y_scaled'], np.array(test_data['y_scaled'])[dev_idx]], axis=0), dtype=torch.float)
    Y_dev = torch.tensor(np.array(test_data['y_scaled'])[dev_idx], dtype=torch.float)
    Y_test = torch.tensor(np.array(test_data['y_scaled'])[test_idx], dtype=torch.float)

    train_essay_set = torch.tensor(np.concatenate([train_data['prompt_ids'], dev_data['prompt_ids'], np.array(test_data['prompt_ids'])[dev_idx]], axis=0), dtype=torch.long)
    dev_essay_set = torch.tensor(np.array(test_data['prompt_ids'])[dev_idx], dtype=torch.long)
    test_essay_set = torch.tensor(np.array(test_data['prompt_ids'])[test_idx], dtype=torch.long)

    # Create Datasets
    train_dataset = TensorDataset(X_source, Y_train, X_source_linguistic_features, X_source_readability, train_essay_set)
    dev_dataset = TensorDataset(X_dev, Y_dev, X_dev_linguistic_features, X_dev_readability, dev_essay_set)
    test_dataset = TensorDataset(X_target, Y_test, X_target_linguistic_features, X_target_readability, test_essay_set)
    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    print('================================')
    print('X_source_pos: ', X_source.size())
    print('X_source_readability: ', X_source_readability.size())
    print('X_source_ling: ', X_source_linguistic_features.size())
    print('Y_train: ', Y_train.size())
    print('Y_train max: ', torch.max(Y_train))
    print('Y_train min: ', torch.min(Y_train))

    print('================================')
    print('X_dev_pos: ', X_dev.size())
    print('X_dev_readability: ', X_dev_readability.size())
    print('X_dev_ling: ', X_dev_linguistic_features.size())
    print('Y_dev: ', Y_dev.size())
    print('Y_dev max: ', torch.max(Y_dev))
    print('Y_dev min: ', torch.min(Y_dev))

    print('================================')
    print('X_target_pos: ', X_target.size())
    print('X_target_readability: ', X_target_readability.size())
    print('X_target_ling: ', X_target_linguistic_features.size())
    print('Y_test: ', Y_test.size())
    print('Y_test max: ', torch.max(Y_test))
    print('Y_test min: ', torch.min(Y_test))
    print('================================')

    # Create model
    model = PAES(
        max_sentnum,
        max_sentlen,
        X_source_linguistic_features.size(1),
        X_source_readability.size(1),
        pos_vocab,
        args.embed_dim,
        args.cnn_filters,
        args.cnn_kernel_size,
        args.lstm_units,
        args.dropout
    ).to(device)

    # Create loss and optimizer
    MSE_Loss = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    wandb.init(project=args.pj_name, name=args.pj_name+str(test_prompt_id), config=dict(args._get_kwargs()))
    # Train loop
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
    for epoch in range(epochs):
        print(f'Seed: {seed}, Prompt: {test_prompt_id}, Epoch: {epoch+1}/{epochs}')
        # Train the model
        train_loss = train_model(model, train_loader, MSE_Loss, optimizer, device)
        # Evaluate the model on dev set
        dev_results = evaluate_model(model, dev_loader, MSE_Loss, device, attribute_name)
        # Evaluate the model on test set
        test_results = evaluate_model(model, test_loader, MSE_Loss, device, attribute_name)

        if dev_results["qwk"] > best_val_metrics[0]:
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = dev_results[met]
                best_test_metrics[i] = test_results[met]

        wandb.log({
            'train_loss': train_loss,
            'dev_loss': dev_results['loss'],
            'test_loss': test_results['loss'],
            'dev_qwk': dev_results['qwk'],
            'dev_lwk': dev_results['lwk'],
            'dev_corr': dev_results['corr'],
            'dev_rmse': dev_results['rmse'],
            'dev_mae': dev_results['mae'],
            'test_qwk': test_results['qwk'],
            'test_lwk': test_results['lwk'],
            'test_corr': test_results['corr'],
            'test_rmse': test_results['rmse'],
            'test_mae': test_results['mae'],
            'best_test_qwk': best_test_metrics[0],
            'best_test_lwk': best_test_metrics[1],
            'best_test_corr': best_test_metrics[2],
            'best_test_rmse': best_test_metrics[3],
            'best_test_mae': best_test_metrics[4],
        })
    
    wandb.alert(title=args.pj_name, text='Training finished!')
    wandb.finish()

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Training PAES model")
    parser.add_argument('--pj_name', type=str, default='PAES (モデル選択をDVRLで使用する開発データを用いて実施)', help='wandb project name for logging')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=50, help='pos embedding dimension')
    parser.add_argument('--cnn_filters', type=int, default=100, help='number of cnn filters')
    parser.add_argument('--cnn_kernel_size', type=int, default=5, help='cnn kernel size')
    parser.add_argument('--lstm_units', type=int, default=100, help='number of lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_type', type=str, default='normal', help='type of model to train', choices=['normal', 'tiny'])
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)