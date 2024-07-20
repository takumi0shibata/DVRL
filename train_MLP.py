'''
This script trains MLP model.
    - for DeBERTa-v3-large embedding vectors
    - for mannualy designed features
'''

import argparse
import numpy as np
from transformers import AutoConfig
import wandb
import torch
import warnings
warnings.filterwarnings('ignore')

from utils.general_utils import set_seed
from utils.load_data import load_data_DVRL, load_data_PAES
from utils.dvrl_utils import (
    remove_top_p_sample,
    fit_func, pred_func,
    calc_qwk,
)
from dvrl.predictor import MLP
from models.features import FeaturesModel
from sklearn.metrics import mean_squared_error


def train_and_evaluate(
        data,
        target_prompt_id,
        weights,
        model_name,
        batch_size,
        epochs,
        device,
        attribute_name,
        valuation_method,
):
    
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    if 'word' in valuation_method:
        config = AutoConfig.from_pretrained(model_name)
        pred_model = MLP(config.hidden_size)
    elif 'pos' in valuation_method:
        pred_model = FeaturesModel()

    fit_func(
        pred_model,
        data['x_source'][weights],
        data['y_source'][weights],
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    y_pred = pred_func(
        pred_model,
        data['x_target'],
        batch_size=batch_size,
        device=device
    )
    y_dev_pred = pred_func(
        pred_model,
        data['x_dev'],
        batch_size=batch_size,
        device=device
    )
    qwk = calc_qwk(data['y_target'], y_pred, target_prompt_id, attribute_name)
    dev_mse = mean_squared_error(data['y_dev'], y_dev_pred)
    return qwk, dev_mse

def main(args):
    target_prompt_id = args.target_prompt_id
    seed = args.seed
    set_seed(seed)
    estimated_data_value = np.load(f'outputs/Estimated_Data_Values/{args.valuation_method}/estimated_data_value{target_prompt_id}.npy')

    ###################################################
    # Training MLP
    ###################################################
    if 'word' in args.valuation_method:
        data = load_data_DVRL(
            f'data/cross_prompt_attributes/{target_prompt_id}/',
            args.attribute_name,
            args.embedding_model,
            args.device,
            devsize=args.dev_size,
        )
    elif 'pos' in args.valuation_method:
        data = load_data_PAES(
            f'data/cross_prompt_attributes/{target_prompt_id}/',
            args.attribute_name,
            args.embedding_model,
            args.device,
            devsize=args.dev_size,
        )
        data['x_source'] = data['x_source'][1]
        data['x_dev'] = data['x_dev'][1]
        data['x_target'] = data['x_target'][1]

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name+str(target_prompt_id),
            config=args
        )
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ##################################################
        # データの価値が低いものを削除
        ##################################################
        set_seed(seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=False,
        )
        qwk_high, dev_loss_high = train_and_evaluate(
            data,
            target_prompt_id,
            weights,
            args.embedding_model,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args.valuation_method,
        )

        ##################################################
        # データの価値が高いものを削除
        ##################################################
        set_seed(seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=True,
        )
        qwk_low, dev_loss_low = train_and_evaluate(
            data,
            target_prompt_id,
            weights,
            args.embedding_model,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args.valuation_method,
        )

        print(f'p: {p_val:.1f}, QWK[High]: {qwk_high:.3f}, QWK[Low]: {qwk_low:.3f}, Dev Loss[High]: {dev_loss_high:.10f}, Dev Loss[Low]: {dev_loss_low:.10f}')

        if args.wandb:
            wandb.log({
                'p': p_val,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
            })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='LOO')
    parser.add_argument('--run_name', type=str, default='train-MLP')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--valuation_method',
        default='DVRL-word',
        choices=[
            'DVRL-word',
            'LOO-word',
            'DataShapley-word',
            'DVRL-pos',
            'LOO-pos',
            'DataShapley-pos',
        ],
    )
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)