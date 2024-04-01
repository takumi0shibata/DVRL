"""Evaluation functions for PAES model."""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error
from utils.general_utils import get_min_max_scores

# 訓練関数の定義
def train_model(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: nn.Module,
        device: torch.device,
        scheduler: nn.Module = None,
        weight: bool = False
        ) -> float:
    """
    Train the model.
    Args:
        model: Model to train
        data_loader: Data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to run the model
        scheduler: Learning rate scheduler
    Returns:
        float: Loss value
    """
    model.train()

    losses = []
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)
    if weight:
        for x_train, y_train, linguistic_train, readability_train, _, weight in progress_bar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            linguistic_train = linguistic_train.to(device)
            readability_train = readability_train.to(device)
    
            # predict
            y_pred = model(x_train, linguistic_train, readability_train)
    
            # calculate loss
            loss = loss_fn(y_pred.squeeze(), y_train.squeeze()) * weight.to(device)
            loss = loss.mean()
                
            losses.append(loss.item())
            loss.backward()
    
            # update weights
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
    
            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})
        
    else:
        for x_train, y_train, linguistic_train, readability_train, _ in progress_bar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            linguistic_train = linguistic_train.to(device)
            readability_train = readability_train.to(device)
    
            # predict
            y_pred = model(x_train, linguistic_train, readability_train)
            
            loss = loss_fn(y_pred.squeeze(), y_train.squeeze())
            losses.append(loss.item())
            loss.backward()
    
            # update weights
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
    
            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

    return np.mean(losses)

# 評価関数の定義
def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        attribute: str,
        average_weight: list = None
        ) -> dict:
    """
    Evaluate the model.
    Args:
        model: Model to evaluate
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to run the model
        attribute: Attribute to evaluate
        average_weight: QWKを計算する際の加重平均用の重み
    Returns:
        dict: Evaluation results
    """
    
    model.eval()

    losses = []
    y_pred_list = []
    y_true_list = []
    essay_set_list = []

    progress_bar = tqdm(data_loader, desc="Evaluation", unit="batch", ncols=100)

    with torch.no_grad():
        for x_input, y_true, linguistic, readability, essay_set in progress_bar:
            y_true = y_true.squeeze()
            if y_true.dim() == 0:
                y_true_list.append(y_true.item())
            else:
                y_true_list.extend(y_true.tolist())
            
            essay_set = essay_set.squeeze()
            if essay_set.dim() == 0:
                essay_set_list.append(essay_set.item()) 
            else:
                essay_set_list.extend(essay_set.tolist())
            x_input = x_input.to(device)
            y_true = y_true.to(device)
            linguistic = linguistic.to(device)
            readability = readability.to(device)

            # predict
            y_pred = model(x_input, linguistic, readability)

            # calculate loss
            loss = loss_fn(y_pred.squeeze(), y_true.squeeze())
            loss = loss.mean()
            losses.append(loss.item())

            squeezed_outputs = y_pred.squeeze()
            y_pred_list.extend(squeezed_outputs.tolist() if squeezed_outputs.dim() > 0 else [squeezed_outputs.item()])
            
            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})


    rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
    mae = mean_absolute_error(y_true_list, y_pred_list)
    corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]

    qwks = []
    lwks = []
    lens = []
    for i in range(1, 9):
        minscore, maxscore = get_min_max_scores()[i][attribute]
        indices = np.where(np.array(essay_set_list) == i)[0]
        if len(indices) > 0:
            rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(y_true_list)[indices])
            rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(y_pred_list)[indices])
            qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
            lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')
            qwks.append(qwk)
            lwks.append(lwk)
            lens.append(len(indices))
    print(np.array(qwks))
    if average_weight is None:
        avg_qwk = np.average(np.array(qwks), weights=lens)
        avg_lwk = np.average(np.array(lwks), weights=lens)
    else:
        avg_qwk = np.average(np.array(qwks), weights=average_weight)
        avg_lwk = np.average(np.array(lwks), weights=average_weight)
    return {
        'loss': np.mean(losses),
        'qwk': avg_qwk,
        'lwk': avg_lwk,
        'corr': corr,
        'rmse': rmse,
        'mae': mae,
        'y_pred': y_pred_list
    }

# 訓練関数の定義
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, use_weight=True):
    model.train()

    losses = []
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)
    for d in progress_bar:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['score'].to(device)
        if use_weight:
            weight = d['weights'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if use_weight:
            loss = loss_fn(outputs.squeeze(), targets) * weight
        else:
            loss = loss_fn(outputs.squeeze(), targets)

        loss = loss.mean()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

    return np.mean(losses)

# 評価関数の定義
def evaluate_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        attribute: str,
        average_weight: list = None
        ) -> dict:
    """
    Evaluate the model.
    Args:
        model: Model to evaluate
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to run the model
        attribute: Attribute to evaluate
    Returns:
        dict: Evaluation results
    """
    
    model.eval()

    losses = []
    y_pred_list = []
    y_true_list = []
    essay_set_list = []

    progress_bar = tqdm(data_loader, desc="Evaluation", unit="batch", ncols=100)

    with torch.no_grad():
        for d in progress_bar:
            # スカラー値をリストとして追加
            scores = d['score'].squeeze()
            if scores.dim() == 0:  # スカラー値の場合
                y_true_list.append(scores.item())  # スカラー値を直接追加
            else:
                y_true_list.extend(scores.tolist())  # リストとして追加
            
            # 以下のコードも同様の問題が発生する可能性があるため、チェックが必要です
            prompts = d['prompt'].squeeze()
            if prompts.dim() == 0:  # スカラー値の場合
                essay_set_list.append(prompts.item())  # スカラー値を直接追加
            else:
                essay_set_list.extend(prompts.tolist())  # リストとして追加
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['score'].to(device)

            # predict
            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)

            # calculate loss
            loss = loss_fn(y_pred.squeeze(), targets.squeeze())
            loss = loss.mean()
            losses.append(loss.item())

            squeezed_outputs = y_pred.squeeze()
            y_pred_list.extend(squeezed_outputs.tolist() if squeezed_outputs.dim() > 0 else [squeezed_outputs.item()])
            
            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})


    rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
    mae = mean_absolute_error(y_true_list, y_pred_list)
    corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]

    qwks = []
    lwks = []
    lens = []
    for i in range(1, 9):
        minscore, maxscore = get_min_max_scores()[i][attribute]
        indices = np.where(np.array(essay_set_list) == i)[0]
        if len(indices) > 0:
            rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(y_true_list)[indices])
            rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(y_pred_list)[indices])
            qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
            lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')
            qwks.append(qwk)
            lwks.append(lwk)
            lens.append(len(indices))
    print(np.array(qwks))
    if average_weight is None:
        avg_qwk = np.average(np.array(qwks), weights=lens)
        avg_lwk = np.average(np.array(lwks), weights=lens)
    else:
        avg_qwk = np.average(np.array(qwks), weights=average_weight)
        avg_lwk = np.average(np.array(lwks), weights=average_weight)
    return {
        'loss': np.mean(losses),
        'qwk': avg_qwk,
        'lwk': avg_lwk,
        'corr': corr,
        'rmse': rmse,
        'mae': mae
    }