import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.evaluation import train_epoch, evaluate_epoch
from models.AES import BERT_Regressor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils.create_embedding_feautres import load_data, normalize_scores, create_data_loader, create_embedding_features
from utils.dvrl_utils import get_dev_sample
import argparse
import wandb


def main(args):
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    output_path = args.output_dir + args.experiment_name + '/'
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = args.max_length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    wandb.init(project=args.wandb_pjname, name=f'BERT-fullsource{args.dev_size}-{test_prompt_id}', config=dict(args._get_kwargs()))

    # Load data
    data_path = args.data_dir + str(test_prompt_id) + '/'
    data = load_data(data_path)

    _, _, test_data = create_embedding_features(data_path, attribute_name, args.embedding_model, device)
    _, _, _, _, dev_idx, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

    train_features = np.concatenate([data['train']['feature'], data['dev']['feature']])
    train_labels = np.concatenate([data['train']['label'], data['dev']['label']])
    train_prompts = np.concatenate([data['train']['essay_set'], data['dev']['essay_set']])
    train_normalized_labels = normalize_scores(train_labels, train_prompts, attribute_name)

    features = np.array(data['test']['feature'])
    labels = np.array(data['test']['label'])
    prompts = np.array(data['test']['essay_set'])
    normalized_labels = normalize_scores(labels, prompts, attribute_name)

    sample_id = dev_idx
    not_sample_id = np.array([i for i in range(len(features)) if i not in sample_id])

    train_data = {}
    dev_data = {}
    test_data = {}

    train_data['feature'] = train_features
    train_data['normalized_label'] = train_normalized_labels
    train_data['essay_set'] = train_prompts

    dev_data['feature'] = features[sample_id]
    dev_data['normalized_label'] = normalized_labels[sample_id]
    dev_data['essay_set'] = prompts[sample_id]

    test_data['feature'] = features[not_sample_id]
    test_data['normalized_label'] = normalized_labels[not_sample_id]
    test_data['essay_set'] = prompts[not_sample_id]


    print('================================')
    print(f'x_train shape: {train_data["feature"].shape}')
    print(f'y_train shape: {train_data["normalized_label"].shape}')
    print('================================')
    print(f'x_dev shape: {dev_data["feature"].shape}')
    print(f'y_dev shape: {dev_data["normalized_label"].shape}')
    print('================================')
    print(f'x_test shape: {test_data["feature"].shape}')
    print(f'y_test shape: {test_data["normalized_label"].shape}')
    print('================================')

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    config = AutoConfig.from_pretrained(model_name)

    # Create data loaders
    train_loader = create_data_loader(train_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
    dev_loader = create_data_loader(dev_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
    test_loader = create_data_loader(test_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)

    # Initialize the model
    model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)

    # Define loss function, optimizer, and scheduler
    loss_fn = nn.MSELoss(reduction='none').to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

    # Training loop
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
    best_dev_loss = 1000
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # Training Set
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
        # Development Set
        dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, attribute_name)
        # Test Set
        eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
    
        if dev_history["qwk"] > best_val_metrics[0]:
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = dev_history[met]
                best_test_metrics[i] = eval_history[met]

        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            'dev_loss': dev_history['loss'],
            'test_loss': eval_history['loss'],
            'dev_qwk': dev_history['qwk'],
            'dev_lwk': dev_history['lwk'],
            'dev_corr': dev_history['corr'],
            'dev_rmse': dev_history['rmse'],
            'dev_mae': dev_history['mae'],
            'test_qwk': eval_history['qwk'],
            'test_lwk': eval_history['lwk'],
            'test_corr': eval_history['corr'],
            'test_rmse': eval_history['rmse'],
            'test_mae': eval_history['mae'],
            'best_qwk': best_test_metrics[0],
            })

    wandb.alert(title=args.wandb_pjname, text='Training finished!')
    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="BERT")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory')
    parser.add_argument('--experiment_name', type=str, default='DVRL_DomainAdaptation', help='name of the experiment')
    parser.add_argument('--dev_size', type=int, default=30, help='size of the dev set')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--wandb_pjname', type=str, default='BERT-fullsource', help='name of the wandb project')
    parser.add_argument('--max_length', type=int, default=512, help='max length of the input')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)