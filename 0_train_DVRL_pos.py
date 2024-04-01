"""Training on DVRL class"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

from dvrl import dvrl_pos
from utils.dvrl_utils import calc_qwk, get_dev_sample
from utils.create_embedding_feautres import create_embedding_features
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, set_seed, pad_text_sequences, flatten_hierarchical_sequences, pad_hierarchical_text_sequences
from utils.general_utils import set_seed
from models.paes import tinyPAES, PAES


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    save_dir = args.output_dir + args.experiment_name + '/'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)
    set_seed(seed)

    data_path = args.data_dir
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': args.features_path,
        'readability_path': args.readability_path
    }

    ###################################################
    # Step1. Create/Load Text Embedding
    ###################################################
    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    x_source_embedding, y_source_embedding = np.concatenate([train_data['essay'], val_data['essay']]), np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    # split test data into dev and test
    _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)
    np.save(save_dir + 'dev_ids.npy', dev_idx)

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
    X_source = torch.tensor(np.concatenate([X_train_pos, X_dev_pos], axis=0), dtype=torch.long)
    X_dev = torch.tensor(X_test_pos[dev_idx], dtype=torch.long)
    X_target = torch.tensor(X_test_pos[test_idx], dtype=torch.long)

    X_source_linguistic_features = torch.tensor(np.concatenate([train_data['features_x'], dev_data['features_x']], axis=0), dtype=torch.float)
    X_dev_linguistic_features = torch.tensor(np.array(test_data['features_x'])[dev_idx], dtype=torch.float)
    X_target_linguistic_features = torch.tensor(np.array(test_data['features_x'])[test_idx], dtype=torch.float)

    X_source_readability = torch.tensor(np.concatenate([train_data['readability_x'], dev_data['readability_x']], axis=0), dtype=torch.float)
    X_dev_readability = torch.tensor(np.array(test_data['readability_x'])[dev_idx], dtype=torch.float)
    X_target_readability = torch.tensor(np.array(test_data['readability_x'])[test_idx], dtype=torch.float)

    Y_source = torch.tensor(np.concatenate([train_data['y_scaled'], dev_data['y_scaled']], axis=0), dtype=torch.float)
    Y_dev = torch.tensor(np.array(test_data['y_scaled'])[dev_idx], dtype=torch.float)
    Y_target = torch.tensor(np.array(test_data['y_scaled'])[test_idx], dtype=torch.float)

    source_essay_set = torch.tensor(np.concatenate([train_data['prompt_ids'], dev_data['prompt_ids']], axis=0), dtype=torch.long)
    dev_essay_set = torch.tensor(np.array(test_data['prompt_ids'])[dev_idx], dtype=torch.long)
    target_essay_set = torch.tensor(np.array(test_data['prompt_ids'])[test_idx], dtype=torch.long)

    X_source_set = (X_source, X_source_linguistic_features, X_source_readability, source_essay_set)
    X_dev_set = (X_dev, X_dev_linguistic_features, X_dev_readability, dev_essay_set)
    X_target_set = (X_target, X_target_linguistic_features, X_target_readability, target_essay_set)


    # print info
    print('================================')
    print('X_source_embedding: ', x_source_embedding.shape)
    print('X_source: ', X_source.shape)
    print('X_source_linguistic_features: ', X_source_linguistic_features.shape)
    print('X_source_readability: ', X_source_readability.shape)
    print('Y_source: ', Y_source.shape)
    print('Y_source max: ', torch.max(Y_source))
    print('Y_source min: ', torch.min(Y_source))

    print('================================')
    print('X_dev: ', X_dev.shape)
    print('X_dev_linguistic_features: ', X_dev_linguistic_features.shape)
    print('X_dev_readability: ', X_dev_readability.shape)
    print('Y_dev: ', Y_dev.shape)
    print('Y_dev max: ', torch.max(Y_dev))
    print('Y_dev min: ', torch.min(Y_dev))

    print('================================')
    print('X_target: ', X_target.shape)
    print('X_target_linguistic_features: ', X_target_linguistic_features.shape)
    print('X_target_readability: ', X_target_readability.shape)
    print('Y_target: ', Y_target.shape)
    print('Y_target max: ', torch.max(Y_target))
    print('Y_target min: ', torch.min(Y_target))
    print('================================')


    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    # 使用可能なGPUの数を取得
    ngpus = torch.cuda.device_count()

    if args.model_type == 'normal':
        pred_model =  PAES(
            max_sentnum,
            max_sentlen,
            X_source_linguistic_features.size(1),
            X_source_readability.size(1), 
            pos_vocab,
            embed_dim=args.embed_dim,
            cnn_filters=args.cnn_filters,
            cnn_kernel_size=args.cnn_kernel_size,
            lstm_units=args.lstm_units,
            dropout=args.dropout
        )
        if ngpus > 1:
            pred_model = nn.DataParallel(pred_model).to(device)
        else:
            pred_model = pred_model.to(device)

    elif args.model_type == 'tiny':
        pred_model =  tinyPAES(
            max_sentnum,
            max_sentlen,
            X_source_linguistic_features.size(1),
            X_source_readability.size(1), 
            pos_vocab,
            embed_dim=args.embed_dim,
            cnn_filters=args.cnn_filters,
            cnn_kernel_size=args.cnn_kernel_size,
            lstm_units=args.lstm_units,
            dropout=args.dropout
        )
        if ngpus > 1:
            pred_model = nn.DataParallel(pred_model).to(device)
        else:
            pred_model = pred_model.to(device)

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
    dvrl_params['inner_iterations'] = 50
    dvrl_params['batch_size_predictor'] = 512
    dvrl_params['moving_average_window'] = 10
    dvrl_params['moving_average'] = False
    dvrl_params['std_penalty_weight'] = None

    # Init wandb
    wandb.init(project=args.wandb_pjname, name=args.experiment_name+str(test_prompt_id), config=dict(args._get_kwargs())|dvrl_params)

    # Initialize DVRL
    dvrl_class = dvrl_pos.Dvrl(X_source_set, Y_source, X_dev_set, Y_dev, pred_model, dvrl_params, device, test_prompt_id, x_source_embedding)

    # Train DVRL
    print('Training DVRL...')
    data_value = dvrl_class.train_dvrl(args.metric)
    np.save(save_dir + f'estimated_data_value{test_prompt_id}.npy', data_value)

    # Pridicts with DVRl
    qwk = dvrl_class.dvrl_predict(X_target_set, Y_target)
    print('Finished data valuation.')
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
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--wandb_pjname', type=str, default='DVRL-pos-本番', help='name of the wandb project')
    parser.add_argument('--device', type=str, default='cuda', help='device to be used', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--embed_dim', type=int, default=50, help='pos embedding dimension')
    parser.add_argument('--cnn_filters', type=int, default=100, help='number of cnn filters')
    parser.add_argument('--cnn_kernel_size', type=int, default=5, help='cnn kernel size')
    parser.add_argument('--lstm_units', type=int, default=100, help='number of lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_type', type=str, default='normal', help='type of model to train', choices=['normal', 'tiny'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)