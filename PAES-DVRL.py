import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
from sklearn.model_selection import train_test_split

from utils.dvrl_utils import get_dev_sample, remove_top_p_sample
from utils.create_embedding_feautres import create_embedding_features
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, set_seed, pad_text_sequences, flatten_hierarchical_sequences, pad_hierarchical_text_sequences
from utils.evaluation import train_model, evaluate_model
from models.paes import tinyPAES, PAES


def main(args):
    test_prompt_id = args.test_prompt_id
    data_value_path = 'outputs/Estimated_Data_Values/PAES/'
    seed = args.seed
    set_seed(seed)
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    attribute_name = args.attribute_name
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

    ########################################################
    # get dev and test indices
    _, _, test_data = create_embedding_features(args.data_dir + str(test_prompt_id) + '/', attribute_name, args.embedding_model, device)
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
    X_target = torch.tensor(X_test_pos[test_idx], dtype=torch.long)

    X_source_linguistic_features = torch.tensor(np.concatenate([train_data['features_x'], dev_data['features_x']], axis=0), dtype=torch.float)
    X_target_linguistic_features = torch.tensor(np.array(test_data['features_x'])[test_idx], dtype=torch.float)

    X_source_readability = torch.tensor(np.concatenate([train_data['readability_x'], dev_data['readability_x']], axis=0), dtype=torch.float)
    X_target_readability = torch.tensor(np.array(test_data['readability_x'])[test_idx], dtype=torch.float)

    Y_source = torch.tensor(np.concatenate([train_data['y_scaled'], dev_data['y_scaled']], axis=0), dtype=torch.float)
    Y_target = torch.tensor(np.array(test_data['y_scaled'])[test_idx], dtype=torch.float)

    source_essay_set = torch.tensor(np.concatenate([train_data['prompt_ids'], dev_data['prompt_ids']], axis=0), dtype=torch.long)
    target_essay_set = torch.tensor(np.array(test_data['prompt_ids'])[test_idx], dtype=torch.long)

    X_source_set = (X_source, X_source_linguistic_features, X_source_readability, source_essay_set)
    X_target_set = (X_target, X_target_linguistic_features, X_target_readability, target_essay_set)


    # print info
    print('================================')
    print('X_source: ', X_source.shape)
    print('X_source_linguistic_features: ', X_source_linguistic_features.shape)
    print('X_source_readability: ', X_source_readability.shape)
    print('Y_source: ', Y_source.shape)

    print('================================')
    print('X_target: ', X_target.shape)
    print('X_target_linguistic_features: ', X_target_linguistic_features.shape)
    print('X_target_readability: ', X_target_readability.shape)
    print('Y_target: ', Y_target.shape)
    print('================================')

    # Create predictor
    print('Creating predictor model...')
    def get_model():
        if args.model_type == 'normal':
            return PAES(
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
            ).to(device)
        elif args.model_type == 'tiny':
            return tinyPAES(
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
            ).to(device)

    data_value = np.load(data_value_path + f'estimated_data_value{test_prompt_id}.npy')

    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)
    for p in np.arange(0.0, 1.0, 0.1):
        # データの価値が低いものを削除
        set_seed(seed)
        weights = remove_top_p_sample(data_value, top_p=p, ascending=False)
        weights = (torch.tensor(weights, dtype=torch.float) == 1)
        X_source_set_tmp = (X_source[weights], X_source_linguistic_features[weights], X_source_readability[weights], source_essay_set[weights])
        Y_source_tmp = Y_source[weights]
        pred_model = get_model()
        optimizer = torch.optim.RMSprop(pred_model.parameters(), lr=args.lr)
        MSE_Loss = nn.MSELoss(reduction='mean').to(device)
        
        # Split data into training and validation sets
        train_indices, val_indices = train_test_split(np.arange(len(X_source_set_tmp[0])), test_size=0.2, random_state=seed)
        X_source_set_train = tuple(x[train_indices] for x in X_source_set_tmp)
        Y_source_train = Y_source_tmp[train_indices]
        X_source_set_val = tuple(x[val_indices] for x in X_source_set_tmp)
        Y_source_val = Y_source_tmp[val_indices]

        train_dataset = TensorDataset(X_source_set_train[0], Y_source_train, X_source_set_train[1], X_source_set_train[2], X_source_set_train[3])
        dev_dataset = TensorDataset(X_source_set_val[0], Y_source_val, X_source_set_val[1], X_source_set_val[2], X_source_set_val[3])
        test_dataset = TensorDataset(X_target_set[0], Y_target, X_target_set[1], X_target_set[2], X_target_set[3])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        best_dev_qwk_high = 0
        best_test_qwk_high = 0
        best_dev_loss_high = 1000
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            train_model(pred_model, train_loader, MSE_Loss, optimizer, device)
            dev_results = evaluate_model(pred_model, dev_loader, MSE_Loss, device, attribute_name)
            test_results = evaluate_model(pred_model, test_loader, MSE_Loss, device, attribute_name)
            if dev_results['qwk'] > best_dev_qwk_high:
                best_dev_qwk_high = dev_results['qwk']
                best_test_qwk_high = test_results['qwk']
                best_dev_loss_high = dev_results['loss']

        # データの価値が高いものを削除
        set_seed(seed)
        weights = remove_top_p_sample(data_value, top_p=p, ascending=True)
        weights = (torch.tensor(weights, dtype=torch.float) == 1)
        X_source_set_tmp = (X_source[weights], X_source_linguistic_features[weights], X_source_readability[weights], source_essay_set[weights])
        Y_source_tmp = Y_source[weights]
        pred_model = get_model()
        optimizer = torch.optim.RMSprop(pred_model.parameters(), lr=0.001)
        MSE_Loss = nn.MSELoss(reduction='mean').to(device)
        
        # Split data into training and validation sets
        train_indices, val_indices = train_test_split(np.arange(len(X_source_set_tmp[0])), test_size=0.2, random_state=seed)
        X_source_set_train = tuple(x[train_indices] for x in X_source_set_tmp)
        Y_source_train = Y_source_tmp[train_indices]
        X_source_set_val = tuple(x[val_indices] for x in X_source_set_tmp)
        Y_source_val = Y_source_tmp[val_indices]

        train_dataset = TensorDataset(X_source_set_train[0], Y_source_train, X_source_set_train[1], X_source_set_train[2], X_source_set_train[3])
        dev_dataset = TensorDataset(X_source_set_val[0], Y_source_val, X_source_set_val[1], X_source_set_val[2], X_source_set_val[3])
        test_dataset = TensorDataset(X_target_set[0], Y_target, X_target_set[1], X_target_set[2], X_target_set[3])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        best_dev_qwk_low = 0
        best_test_qwk_low = 0
        best_dev_loss_low = 100
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            train_model(pred_model, train_loader, MSE_Loss, optimizer, device)
            dev_results = evaluate_model(pred_model, dev_loader, MSE_Loss, device, attribute_name)
            test_results = evaluate_model(pred_model, test_loader, MSE_Loss, device, attribute_name)
            if dev_results['qwk'] > best_dev_qwk_low:
                best_dev_qwk_low = dev_results['qwk']
                best_test_qwk_low = test_results['qwk']
                best_dev_loss_low = dev_results['loss']

        wandb.log({
            'p': p,
            'best_dev_qwk_high': best_dev_qwk_high,
            'best_test_qwk_high': best_test_qwk_high,
            'best_dev_loss_high': best_dev_loss_high,
            'best_dev_qwk_low': best_dev_qwk_low,
            'best_test_qwk_low': best_test_qwk_low,
            'best_dev_loss_low': best_dev_loss_low
        })
    
    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='DVRL-PAES', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--model_type', type=str, default='normal', help='type of model to train', choices=['normal', 'tiny'])
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=50, help='pos embedding dimension')
    parser.add_argument('--cnn_filters', type=int, default=100, help='number of cnn filters')
    parser.add_argument('--cnn_kernel_size', type=int, default=5, help='cnn kernel size')
    parser.add_argument('--lstm_units', type=int, default=100, help='number of lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)