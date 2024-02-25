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
    for x_train, y_train, linguistic_train, readability_train, _, weight in progress_bar:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        linguistic_train = linguistic_train.to(device)
        readability_train = readability_train.to(device)

        # predict
        y_pred = model(x_train, linguistic_train, readability_train)

        # calculate loss
        if weight is not None:
            loss = loss_fn(y_pred.squeeze(), y_train.squeeze()) * weight.to(device)
            loss = loss.mean()
        else:
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
        attribute: str
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
        for x_input, y_true, linguistic, readability, essay_set in progress_bar:
            y_true_list.extend(y_true.squeeze().tolist())
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
    return {
        'loss': np.mean(losses),
        'qwk': np.average(np.array(qwks), weights=lens),
        'lwk': np.average(np.array(lwks), weights=lens),
        'corr': corr,
        'rmse': rmse,
        'mae': mae
    }

# 訓練関数の定義
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()

    losses = []
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)
    for d in progress_bar:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['score'].to(device)
        weight = d['weights'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs.squeeze(), targets) * weight
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
        attribute: str
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
            y_true_list.extend(d['score'].squeeze().tolist())
            essay_set_list.extend(d['prompt'].squeeze().tolist())
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
    return {
        'loss': np.mean(losses),
        'qwk': np.average(np.array(qwks), weights=lens),
        'lwk': np.average(np.array(lwks), weights=lens),
        'corr': corr,
        'rmse': rmse,
        'mae': mae
    }