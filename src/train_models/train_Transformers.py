'''
This script is used to train the model using the Transformer-based model.
    - BERT
    - DeBERTa-v3-large
'''
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
import wandb
from dvrl.dataset import EssayDataset

# my packages
from utils.dvrl_utils import remove_top_p_sample 
from utils.create_embedding_feautres import create_data_loader
from utils.evaluation import train_epoch, evaluate_epoch
from models.transfomer_enc import BERT_Regressor
from utils.general_utils import set_seed
from utils.load_data import load_data_Transformers


def train_and_evaluate(
    data,
    p: float,
    seed: int,
    estimated_data_value: np.ndarray,
    attribute_name: str,
    device: str,
    EPOCHS: int,
    MAX_LEN: int,
    BATCH_SIZE: int,
    ascending=False
):
    set_seed(seed)
    weights = remove_top_p_sample(
        estimated_data_value,
        top_p=p,
        ascending=ascending,
    )
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    source_data = {
        'feature': data['x_source'][weights],
        'normalized_label': data['y_source'][weights],
        'essay_set': data['source_prompts'][weights],
    }

    dev_data = {
        'feature': data['x_dev'],
        'normalized_label': data['y_dev'],
        'essay_set': data['dev_prompts'],
    }

    target_data = {
        'feature': data['x_target'],
        'normalized_label': data['y_target'],
        'essay_set': data['target_prompts'],
    }
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    config = AutoConfig.from_pretrained(model_name)
    
    train_loader = create_data_loader(source_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
    dev_loader = create_data_loader(dev_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
    test_loader = create_data_loader(target_data, tokenizer, max_length=MAX_LEN, batch_size=BATCH_SIZE)
    
    # Initialize the model
    model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)
    
    # Define loss function, optimizer, and scheduler
    loss_fn = nn.MSELoss(reduction='none').to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=len(train_loader)*EPOCHS
    )
    
    # Training loop
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
    best_dev_loss = 1000
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
        dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, attribute_name)
        eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
    
        if dev_history["qwk"] > best_val_metrics[0]:
            best_dev_loss = dev_history['loss']
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = dev_history[met]
                best_test_metrics[i] = eval_history[met]

    return best_val_metrics, best_test_metrics, best_dev_loss



def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_lambda{args.loss_lambda}_ot{args.ot}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        dev_size=args.dev_size,
        cache_dir='src/.embedding_cache',
        embedding_model=args.embedding_model,
        add_pos=False,
    )
    estimated_data_value = np.load(f'outputs/dvrl_v5/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}.npy')
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')

    
    for p in np.arange(0.0, 1.0, 0.1):
        data = load_data_Transformers(
            f'data/cross_prompt_attributes/{target_prompt_id}/',
            args.attribute_name,
            args.embedding_model,
            args.device,
            devsize=args.dev_size
        )

        ###################################################
        # データの価値が低いものを削除
        ###################################################
        best_val_metrics_high, best_test_metrics_high, best_dev_loss_high = train_and_evaluate(
            data,
            p,
            args.seed,
            estimated_data_value,
            args.attribute_name,
            args.device,
            args.epochs,
            args.max_length,
            args.batch_size,
            ascending=False
        )
        ###################################################
        # データの価値が高いものを削除
        ###################################################
        best_val_metrics_low, best_test_metrics_low, best_dev_loss_low = train_and_evaluate(
            data,
            p,
            args.seed,
            estimated_data_value,
            args.attribute_name,
            args.device,
            args.epochs,
            args.max_length,
            args.batch_size,
            ascending=True
        )
        
        if args.wandb:
            wandb.log({
                'p': p,
                'Dev QWK [high]': best_val_metrics_high[0],
                'Test QWK [high]': best_test_metrics_high[0],
                'Dev Loss [high]': best_dev_loss_high,
                'Dev QWK [low]': best_val_metrics_low[0],
                'Test QWK [low]': best_test_metrics_low[0],
                'Dev Loss [low]': best_dev_loss_low
            })
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser('Training Model')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL', choices=['DVRL', 'LOO', 'DataShapley'])
    parser.add_argument('--run_name', type=str, default='train-BERT')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--max_length', type=int, default=512, choices=[512, 256]) # BERT-base: 512, DeBERTa-v3-large: 256
    parser.add_argument('--batch_size', type=int, default=16, choices=[16, 8]) # BERT-base: 16, DeBERTa-v3-large: 8
    parser.add_argument('--epochs', type=int, default=10, choices=[10, 5]) # BERT-base: 10, DeBERTa-v3-large: 5
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--ot', action='store_true')
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased',
        choices=[
            'bert-base-uncased',
            'microsoft/deberta-v3-large'
        ]
    )
    parser.add_argument(
        '--valuation_method',
        default='DVRL-word',
        choices=[
            'DVRL-word',
            'LOO-word',
            'DataShapley-word',
        ],
    )

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)