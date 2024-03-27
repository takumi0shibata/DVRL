"""This script trains the PAES_on_torch model on the given prompt and attribute."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

# import my modules
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences, set_seed, pad_text_sequences, flatten_hierarchical_sequences
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

    if args.model_type == 'normal':
        # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
        X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
        X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
        X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

        X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
        X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
        X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))
    elif args.model_type == 'tiny':
        X_train_pos = flatten_hierarchical_sequences(train_data['pos_x'])
        X_dev_pos = flatten_hierarchical_sequences(dev_data['pos_x'])
        X_test_pos = flatten_hierarchical_sequences(test_data['pos_x'])

        max_length = max(max([len(x) for x in X_train_pos]), max([len(x) for x in X_dev_pos]), max([len(x) for x in X_test_pos]))

        X_train_pos = pad_text_sequences(X_train_pos, max_length)
        X_dev_pos = pad_text_sequences(X_dev_pos, max_length)
        X_test_pos = pad_text_sequences(X_test_pos, max_length)

        if max_length > 512:
            X_train_pos = X_train_pos[:, :512]
            X_dev_pos = X_dev_pos[:, :512]
            X_test_pos = X_test_pos[:, :512]


    # convert to tensor
    X_train = torch.tensor(X_train_pos, dtype=torch.long)
    X_dev = torch.tensor(X_dev_pos, dtype=torch.long)
    X_test = torch.tensor(X_test_pos, dtype=torch.long)

    X_train_linguistic_features = torch.tensor(np.array(train_data['features_x']), dtype=torch.float)
    X_dev_linguistic_features = torch.tensor(np.array(dev_data['features_x']), dtype=torch.float)
    X_test_linguistic_features = torch.tensor(np.array(test_data['features_x']), dtype=torch.float)

    X_train_readability = torch.tensor(np.array(train_data['readability_x']), dtype=torch.float)
    X_dev_readability = torch.tensor(np.array(dev_data['readability_x']), dtype=torch.float)
    X_test_readability = torch.tensor(np.array(test_data['readability_x']), dtype=torch.float)

    Y_train = torch.tensor(np.array(train_data['y_scaled']), dtype=torch.float)
    Y_dev = torch.tensor(np.array(dev_data['y_scaled']), dtype=torch.float)
    Y_test = torch.tensor(np.array(test_data['y_scaled']), dtype=torch.float)

    train_essay_set = torch.tensor(np.array(train_data['prompt_ids']), dtype=torch.long)
    dev_essay_set = torch.tensor(np.array(dev_data['prompt_ids']), dtype=torch.long)
    test_essay_set = torch.tensor(np.array(test_data['prompt_ids']), dtype=torch.long)

    # Create Datasets
    train_dataset = TensorDataset(X_train, Y_train, X_train_linguistic_features, X_train_readability, train_essay_set)
    dev_dataset = TensorDataset(X_dev, Y_dev, X_dev_linguistic_features, X_dev_readability, dev_essay_set)
    test_dataset = TensorDataset(X_test, Y_test, X_test_linguistic_features, X_test_readability, test_essay_set)
    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    print('================================')
    print('X_train_pos: ', X_train.size())
    print('X_train_readability: ', X_train_readability.size())
    print('X_train_ling: ', X_train_linguistic_features.size())
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
    print('X_test_pos: ', X_test.size())
    print('X_test_readability: ', X_test_readability.size())
    print('X_test_ling: ', X_test_linguistic_features.size())
    print('Y_test: ', Y_test.size())
    print('Y_test max: ', torch.max(Y_test))
    print('Y_test min: ', torch.min(Y_test))
    print('================================')

    # Create model
    if args.model_type == 'normal':
        model = PAES(
            max_sentnum,
            max_sentlen,
            X_train_linguistic_features.size(1),
            X_train_readability.size(1), 
            pos_vocab,
            args.embed_dim,
            args.cnn_filters,
            args.cnn_kernel_size,
            args.lstm_units,
            args.dropout
            ).to(device)
    elif args.model_type == 'tiny':
        model = tinyPAES(
            max_sentnum,
            max_sentlen,
            X_train_linguistic_features.size(1),
            X_train_readability.size(1), 
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
    parser.add_argument('--pj_name', type=str, default='PAES', help='wandb project name for logging')
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
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)