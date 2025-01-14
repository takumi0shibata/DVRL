import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import polars as pl

from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from models.paes import PAES
from dvrl.predictor_config import PAESModelConfig
from dvrl.fn_predictor import fit_func, pred_func, calc_qwk

for prompt in range(1, 9):
    target_prompt_id = prompt
    device = torch.device('cuda')
    set_seed(12)
    
    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(
        target_prompt_set=target_prompt_id,
        dev_size=30,
        cache_dir='src/.embedding_cache',
        embedding_model='microsoft/deberta-v3-large',
        add_pos=True,
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')
    
    from sklearn.model_selection import train_test_split
    
    train_index = np.array(range(len(train_data['essay_id'])))
    
    # 擬似ラベルを付与するデータとしないデータに分割
    train_index, val_index = train_test_split(
        train_index,
        test_size=0.2,
        random_state=12,
        shuffle=True
    )
    
    config = PAESModelConfig()
    model = PAES(train_data['max_sentnum'], train_data['max_sentlen'], train_data['pos_vocab']).to(device)
    # train data
    x_train = [train_data['pos_x'][train_index], train_data['feature'][train_index], train_data['readability'][train_index]]
    y_train = train_data['scaled_score'][train_index].reshape(-1, 1)
    # dev data
    x_dev = [train_data['pos_x'][val_index], train_data['feature'][val_index], train_data['readability'][val_index]]
    y_dev = train_data['scaled_score'][val_index].reshape(-1, 1)
    # test data
    x_test = [
        np.concatenate([test_data['pos_x'], dev_data['pos_x']], axis=0),
        np.concatenate([test_data['feature'], dev_data['feature']], axis=0),
        np.concatenate([test_data['readability'], dev_data['readability']], axis=0)
    ]
    y_test = np.concatenate([test_data['scaled_score'], dev_data['scaled_score']])
    
    model = fit_func(
        model,
        x_train,
        y_train,
        config.optimizer,
        config.lr,
        config.batch_size,
        config.epochs,
        device,
        target_prompt_id,
        'mse',
        x_dev,
        y_dev,
        False,
        verbose=True
    )
    
    # Predict
    print('Predicting...')
    y_dev_pred = pred_func(
        model,
        x_dev,
        config.batch_size,
        device
    )
    y_test_pred = pred_func(
        model,
        x_test,
        config.batch_size,
        device
    )
    
    # Calculate QWK
    print('Calculating QWK...')
    # dev_qwk = calc_qwk(y_dev, y_dev_pred, target_prompt_id, 'score')
    test_qwk = calc_qwk(y_test, y_test_pred, target_prompt_id, 'score')
    
    # print(f'    Dev QWK: {dev_qwk}')
    print(f'    Test QWK: {test_qwk}')
    
    import polars as pl
    # y_test_predの値をCSVで保存
    df = pl.DataFrame({
        'essay_id': np.concatenate([test_data['essay_id'], dev_data['essay_id']]),
        'y_pred': y_test_pred.flatten()
    })
    df.write_csv(f'outputs/paes/prediction_{target_prompt_id}.csv')