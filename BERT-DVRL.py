import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
import wandb
from sklearn.model_selection import train_test_split

# my packages
from utils.dvrl_utils import remove_top_p_sample, get_dev_sample 
from utils.create_embedding_feautres import load_data, normalize_scores, create_data_loader, create_embedding_features
from utils.evaluation import train_epoch, evaluate_epoch
from models.transfomer_enc import BERT_Regressor
from utils.general_utils import set_seed


def main(args):
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    set_seed(seed)
    device = args.device
    data_path = args.data_dir + str(test_prompt_id) + '/'
    data_value_path = 'outputs/Estimated_Data_Values/MLP/'

    # set parameters
    EPOCHS = args.epochs
    MAX_LEN = args.max_length
    BATCH_SIZE = args.batch_size

    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)
    interval = 0.1
    for p in np.arange(0.0, 1.0, interval):

        # Load data
        data = load_data(data_path)
        _, _, test_data = create_embedding_features(data_path, attribute_name, args.embedding_model, device)
        _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

        train_features = np.concatenate([data['train']['feature'], data['dev']['feature']])
        train_labels = np.concatenate([data['train']['label'], data['dev']['label']])
        train_prompts = np.concatenate([data['train']['essay_set'], data['dev']['essay_set']])
        train_normalized_labels = normalize_scores(train_labels, train_prompts, attribute_name)

        test_features = np.array(data['test']['feature'])
        test_labels = np.array(data['test']['label'])
        test_prompts = np.array(data['test']['essay_set'])
        test_normalized_labels = normalize_scores(test_labels, test_prompts, attribute_name)

        train_data = {}
        test_data = {}

        ###################################################
        # データの価値が低いものを削除
        ###################################################
        set_seed(seed)
        weights = remove_top_p_sample(np.load(data_value_path + f'estimated_data_value{test_prompt_id}.npy'), top_p=p, ascending=False)
        weights = (torch.tensor(weights, dtype=torch.float) == 1)
        train_data['feature'] = train_features[weights]
        train_data['normalized_label'] = train_normalized_labels[weights]
        train_data['essay_set'] = train_prompts[weights]

        # Split train_data into training and validation sets (80% train, 20% validation)
        train_indices, val_indices = train_test_split(np.arange(len(train_data['feature'])), test_size=0.2, random_state=seed)

        # Create validation data
        val_data = {
            'feature': train_data['feature'][val_indices],
            'normalized_label': train_data['normalized_label'][val_indices],
            'essay_set': train_data['essay_set'][val_indices]
        }

        # Update train_data to exclude validation data
        train_data['feature'] = np.concatenate([train_data['feature'][train_indices], test_features[dev_idx]])
        train_data['normalized_label'] = np.concatenate([train_data['normalized_label'][train_indices], test_normalized_labels[dev_idx]])
        train_data['essay_set'] = np.concatenate([train_data['essay_set'][train_indices], test_prompts[dev_idx]])

        test_data['feature'] = test_features[test_idx]
        test_data['normalized_label'] = test_normalized_labels[test_idx]
        test_data['essay_set'] = test_prompts[test_idx]


        print('================================')
        print(f'x_train shape: {train_data["feature"].shape}')
        print(f'y_train shape: {train_data["normalized_label"].shape}')
        print('================================')
        print(f'x_dev shape: {val_data["feature"].shape}')
        print(f'y_dev shape: {val_data["normalized_label"].shape}')
        print('================================')
        print(f'x_test shape: {test_data["feature"].shape}')
        print(f'y_test shape: {test_data["normalized_label"].shape}')
        print('================================')
        
        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        config = AutoConfig.from_pretrained(model_name)
        

        train_loader = create_data_loader(train_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        dev_loader = create_data_loader(val_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        test_loader = create_data_loader(test_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        
        # Initialize the model
        model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)
        
        # Define loss function, optimizer, and scheduler
        loss_fn = nn.MSELoss(reduction='none').to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)
        
        # Training loop
        best_test_metrics_high = [-1, -1, -1, -1, -1]
        best_val_metrics_high = [-1, -1, -1, -1, -1]
        best_dev_loss_high = 1000
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
            dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, attribute_name)
            eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
        
            if dev_history["qwk"] > best_val_metrics_high[0]:
                best_dev_loss_high = dev_history['loss']
                for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                    best_val_metrics_high[i] = dev_history[met]
                    best_test_metrics_high[i] = eval_history[met]

        
        ###################################################
        # データの価値が高いものを削除
        ###################################################
        set_seed(seed)
        weights = remove_top_p_sample(np.load(data_value_path + f'estimated_data_value{test_prompt_id}.npy'), top_p=p, ascending=True)
        weights = (torch.tensor(weights, dtype=torch.float) == 1)
        train_data['feature'] = train_features[weights]
        train_data['normalized_label'] = train_normalized_labels[weights]
        train_data['essay_set'] = train_prompts[weights]

        # Split train_data into training and validation sets (80% train, 20% validation)
        train_indices, val_indices = train_test_split(np.arange(len(train_data['feature'])), test_size=0.2, random_state=seed)

        # Create validation data
        val_data = {
            'feature': train_data['feature'][val_indices],
            'normalized_label': train_data['normalized_label'][val_indices],
            'essay_set': train_data['essay_set'][val_indices]
        }

        # Update train_data to exclude validation data
        train_data['feature'] = np.concatenate([train_data['feature'][train_indices], test_features[dev_idx]])
        train_data['normalized_label'] = np.concatenate([train_data['normalized_label'][train_indices], test_normalized_labels[dev_idx]])
        train_data['essay_set'] = np.concatenate([train_data['essay_set'][train_indices], test_prompts[dev_idx]])

        test_data['feature'] = test_features[test_idx]
        test_data['normalized_label'] = test_normalized_labels[test_idx]
        test_data['essay_set'] = test_prompts[test_idx]


        print('================================')
        print(f'x_train shape: {train_data["feature"].shape}')
        print(f'y_train shape: {train_data["normalized_label"].shape}')
        print('================================')
        print(f'x_dev shape: {val_data["feature"].shape}')
        print(f'y_dev shape: {val_data["normalized_label"].shape}')
        print('================================')
        print(f'x_test shape: {test_data["feature"].shape}')
        print(f'y_test shape: {test_data["normalized_label"].shape}')
        print('================================')
        
        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        config = AutoConfig.from_pretrained(model_name)
        

        train_loader = create_data_loader(train_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        dev_loader = create_data_loader(val_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        test_loader = create_data_loader(test_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
        
        # Initialize the model
        model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)
        
        # Define loss function, optimizer, and scheduler
        loss_fn = nn.MSELoss(reduction='none').to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)
        
        # Training loop
        best_test_metrics_low = [-1, -1, -1, -1, -1]
        best_val_metrics_low = [-1, -1, -1, -1, -1]
        best_dev_loss_low = 1000
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
            dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, attribute_name)
            eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
        
            if dev_history["qwk"] > best_val_metrics_low[0]:
                best_dev_loss_low = dev_history['loss']
                for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                    best_val_metrics_low[i] = dev_history[met]
                    best_test_metrics_low[i] = eval_history[met]
        
        wandb.log({
            'p': p,
            'best_dev_qwk_high': best_val_metrics_high[0],
            'best_test_qwk_high': best_test_metrics_high[0],
            'best_dev_loss_high': best_dev_loss_high,
            'best_dev_qwk_low': best_val_metrics_low[0],
            'best_test_qwk_low': best_test_metrics_low[0],
            'best_dev_loss_low': best_dev_loss_low
            })
    
    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='BERT-DVRL', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--max_length', type=int, default=512, help='max length of the input')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='name of the pre-trained model')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)