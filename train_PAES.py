'''
This script trains a PAES model on the target prompt using the estimated data values.
'''

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

from utils.dvrl_utils import remove_top_p_sample
from utils.evaluation import train_model, evaluate_model
from utils.general_utils import set_seed
from utils.load_data import load_data_PAES
from models.paes import PAES


def train_and_evaluate(
    p: float,
    estimated_data_value: np.ndarray,
    ascending: bool,
    data: dict,
    device: str,
    batch_size: int,
    attribute_name: str,
    epochs: int,
) -> tuple[float, float, float]:
    weights = remove_top_p_sample(
        estimated_data_value,
        top_p=p,
        ascending=ascending
    )
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    pred_model = PAES(
        max_num=data['max_sentnum'],
        max_len=data['max_sentlen'],
        pos_vocab=data['pos_vocab'],
    )
    optimizer = torch.optim.RMSprop(pred_model.parameters(), lr=args.lr)
    MSE_Loss = nn.MSELoss(reduction='mean').to(device)

    source_dataset = TensorDataset(
        torch.tensor(data['x_source'][0], dtype=torch.long)[weights],
        torch.tensor(data['y_source'], dtype=torch.float)[weights],
        torch.tensor(data['x_source_linguistic_features'], dtype=torch.float)[weights],
        torch.tensor(data['x_source_readability'], dtype=torch.float)[weights],
        torch.tensor(data['x_source'][2], dtype=torch.long)[weights],
    )
    dev_dataset = TensorDataset(
        torch.tensor(data['x_dev'][0], dtype=torch.long),
        torch.tensor(data['y_dev'], dtype=torch.float),
        torch.tensor(data['x_dev_linguistic_features'], dtype=torch.float),
        torch.tensor(data['x_dev_readability'], dtype=torch.float),
        torch.tensor(data['x_dev'][2], dtype=torch.long),
    )
    target_dataset = TensorDataset(
        torch.tensor(data['x_target'][0], dtype=torch.long),
        torch.tensor(data['y_target'], dtype=torch.float),
        torch.tensor(data['x_target_linguistic_features'], dtype=torch.float),
        torch.tensor(data['x_target_readability'], dtype=torch.float),
        torch.tensor(data['x_target'][2], dtype=torch.long),
    )
    train_loader = DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)


    best_dev_qwk = 0
    best_test_qwk = 0
    best_dev_loss = 1000
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        train_model(
            pred_model,
            train_loader,
            MSE_Loss,
            optimizer,
            device
        )
        dev_results = evaluate_model(
            pred_model,
            dev_loader,
            MSE_Loss,
            device,
            attribute_name
        )
        test_results = evaluate_model(
            pred_model,
            test_loader,
            MSE_Loss,
            device,
            attribute_name
        )

        if dev_results['qwk'] > best_dev_qwk:
            best_dev_qwk = dev_results['qwk']
            best_test_qwk = test_results['qwk']
            best_dev_loss = dev_results['loss']

    return best_dev_qwk, best_test_qwk, best_dev_loss

def main(args):
    target_prompt_id = args.target_prompt_id
    seed = args.seed
    set_seed(seed)
    estimated_data_value = np.load(f'outputs/Estimated_Data_Values/{args.valuation_method}/estimated_data_value{target_prompt_id}.npy')

    # Load data
    data = load_data_PAES(
        f'data/cross_prompt_attributes/{target_prompt_id}/',
        args.attribute_name,
        args.embedding_model,
        args.device,
        devsize=args.dev_size
    )

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name+str(target_prompt_id),
            config=args
        )

    for p in np.arange(0.0, 1.0, 0.1):
        ################################################
        # データの価値が低いものを削除
        ################################################
        set_seed(seed)
        best_dev_qwk_high, best_test_qwk_high, best_dev_loss_high = train_and_evaluate(
            p,
            estimated_data_value,
            False,
            data,
            args.device,
            args.batch_size,
            args.attribute_name,
            args.epochs,
        )

        ################################################
        # データの価値が高いものを削除
        ################################################
        set_seed(seed)
        best_dev_qwk_low, best_test_qwk_low, best_dev_loss_low = train_and_evaluate(
            p,
            estimated_data_value,
            True,
            data,
            args.device,
            args.batch_size,
            args.attribute_name,
            args.epochs,
        )
        
        if args.wandb:
            wandb.log({
                'p': p,
                'best_dev_qwk_high': best_dev_qwk_high,
                'best_test_qwk_high': best_test_qwk_high,
                'best_dev_loss_high': best_dev_loss_high,
                'best_dev_qwk_low': best_dev_qwk_low,
                'best_test_qwk_low': best_test_qwk_low,
                'best_dev_loss_low': best_dev_loss_low
            })
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL')
    parser.add_argument('--run_name', type=str, default='train-PAES')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument(
        '--valuation_method',
        default='DVRL-pos',
        choices=[
            'DVRL-pos',
            'LOO-pos',
            'DataShapley-pos',
        ],
    )
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)