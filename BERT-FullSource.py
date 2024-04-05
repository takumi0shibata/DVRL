import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
import wandb

# my packages
from utils.evaluation import train_epoch, evaluate_epoch
from models.transfomer_enc import BERT_Regressor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils.create_embedding_feautres import load_data, normalize_scores, create_data_loader, create_embedding_features
from utils.dvrl_utils import get_dev_sample
from utils.general_utils import set_seed


def main(args):
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    set_seed(seed)
    device = args.device
    data_path = args.data_dir + str(test_prompt_id) + '/'

    # parameters
    MAX_LEN = args.max_length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Init wandb
    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)

    # Load data
    data = load_data(data_path)
    _, _, test_data = create_embedding_features(data_path, attribute_name, args.embedding_model, device)
    _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

    features = np.array(data['test']['feature'])
    labels = np.array(data['test']['label'])
    prompts = np.array(data['test']['essay_set'])
    normalized_labels = normalize_scores(labels, prompts, attribute_name)

    train_features = np.concatenate([data['train']['feature'], data['dev']['feature'], features[dev_idx]])
    train_labels = np.concatenate([data['train']['label'], data['dev']['label'], labels[dev_idx]])
    train_prompts = np.concatenate([data['train']['essay_set'], data['dev']['essay_set'], prompts[dev_idx]])
    train_normalized_labels = normalize_scores(train_labels, train_prompts, attribute_name)

    features = np.array(data['test']['feature'])
    labels = np.array(data['test']['label'])
    prompts = np.array(data['test']['essay_set'])
    normalized_labels = normalize_scores(labels, prompts, attribute_name)

    train_data = {}
    dev_data = {}
    test_data = {}

    train_data['feature'] = train_features
    train_data['normalized_label'] = train_normalized_labels
    train_data['essay_set'] = train_prompts

    dev_data['feature'] = features[dev_idx]
    dev_data['normalized_label'] = normalized_labels[dev_idx]
    dev_data['essay_set'] = prompts[dev_idx]

    test_data['feature'] = features[test_idx]
    test_data['normalized_label'] = normalized_labels[test_idx]
    test_data['essay_set'] = prompts[test_idx]


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

    model_name = args.model_name
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
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

    # Training loop
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
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
            'best_lwk': best_test_metrics[1],
            'best_corr': best_test_metrics[2],
            'best_rmse': best_test_metrics[3],
            'best_mae': best_test_metrics[4]
            })

    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='BERT-FullSource', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--max_length', type=int, default=512, help='max length of the input')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='name of the pre-trained model')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)