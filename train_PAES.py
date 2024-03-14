"""This script trains the PAES_on_torch model on the given prompt and attribute."""

import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# import my modules
from configs.configs import Configs
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences
from models.PAES import PAES, fastPAES
from utils.evaluation import train_model, evaluate_model

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="PAES_on_torch model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--device', type=str, default='mac', help='linux or mac')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory')
    parser.add_argument('--experiment_name', type=str, default='fastPAES', help='name of the experiment')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed

    # Set device
    if args.device == 'linux':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mac':
        device = torch.device('mps')
    else:
        raise ValueError("device must be either 'linux' or 'mac'")
    
    # fix random seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Attribute: {}".format(attribute_name))
    print("Seed: {}".format(seed))
    print("Device: {}".format(device))

    # Load configs
    configs = Configs()

    data_path = configs.DATA_PATH3
    print(f'load data from {data_path}...')
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    features_path = configs.FEATURES_PATH
    readability_path = configs.READABILITY_PATH
    vocab_size = configs.VOCAB_SIZE
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
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

    # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
    X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

    X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))

    # convert to tensor
    X_train= torch.tensor(X_train_pos, dtype=torch.long)
    X_dev = torch.tensor(X_dev_pos, dtype=torch.long)
    X_test= torch.tensor(X_test_pos, dtype=torch.long)

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
    # model = PAES_on_torch(100, 100, batch_size, len(pos_vocab), configs, X_train_linguistic_features.size()[1], X_train_readability.size()[1], device)
    # model = PAES(max_sentnum, max_sentlen, X_train_linguistic_features.size(1), X_train_readability.size(1), pos_vocab=pos_vocab)
    model = fastPAES(max_sentnum, max_sentlen, X_train_linguistic_features.size(1), X_train_readability.size(1), pos_vocab=pos_vocab)
    model = model.to(device)
    print(model)

    # Create loss and optimizer
    MSE_Loss = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    train_history = []
    dev_history = []
    test_history = []
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
    for epoch in range(epochs):
        print('{} / {} EPOCHS'.format(epoch+1, epochs))
        print('Seed: {}, Prompt: {}'.format(seed, test_prompt_id))
        
        # Train the model
        train_loss = train_model(model, train_loader, MSE_Loss, optimizer, device)
        print(f'Train loss: {train_loss: .4f}')
        train_history.append(train_loss)

        # Evaluate the model on dev set
        dev_results = evaluate_model(model, dev_loader, MSE_Loss, device, attribute_name)
        print(f'Validation loss: {dev_results["loss"]: .4f}')
        dev_history.append(dev_results["loss"])

        # Evaluate the model on test set
        test_results = evaluate_model(model, test_loader, MSE_Loss, device, attribute_name)
        print(f'Test loss: {test_results["loss"]: .4f}')
        test_history.append(test_results["loss"])

        print(f'[VAL]  -> QWK: {dev_results["qwk"]: .3f}, CORR: {dev_results["corr"]: .3f}, RMSE: {dev_results["rmse"]: .3f}')
        print(f'[TEST] -> QWK: {test_results["qwk"]: .3f}, CORR: {test_results["corr"]: .3f}, RMSE: {test_results["rmse"]: .3f}')

        if dev_results["qwk"] > best_val_metrics[0]:
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = dev_results[met]
                best_test_metrics[i] = test_results[met]

        print(f'[BEST] -> QWK: {best_test_metrics[0]: .3f}, CORR: {best_test_metrics[2]: .3f}, RMSE: {best_test_metrics[3]: .3f}')

    # save metrics
    output_dir = args.output_dir + args.experiment_name + '/' + str(seed) + '/'
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(np.array(best_test_metrics).reshape(1, 5), columns=['qwk', 'lwk', 'corr', 'rmse', 'mae']).to_csv(output_dir + f'best_metrics_prompt{test_prompt_id}.csv', index=False, header=True)

    # plot loss
    epochs = list(range(1, epochs+1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history, label='Train', color='blue')
    plt.plot(epochs, dev_history, label='Validate', color='green')
    plt.plot(epochs, test_history, label='Test', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir + f'loss_prompt{test_prompt_id}.png')

if __name__ == '__main__':
    main()