"""Utility functions for DVRL model."""

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from utils.general_utils import get_min_max_scores


def fit_func(
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        epochs: int,
        device: torch.device,
        sample_weight: np.ndarray = None
        ) -> list:
    """
    Fit the model with the given data.
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to run the model
        sample_weight: Sample weight for each data
    Returns:
        list: Loss history
    """

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    if sample_weight is not None:
        loss_fn = nn.MSELoss(reduction='none')
        sample_weight = torch.tensor(sample_weight, dtype=torch.float)
        train_data = TensorDataset(x_train, y_train, sample_weight)
    else:
        loss_fn = nn.MSELoss(reduction='mean')
        train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)

    history = []
    for epoch in range(epochs):
        losses = []
        if sample_weight is not None:
            for x_batch, y_batch, w_batch in train_loader:
                optimizer.zero_grad()
                x_batch, y_batch, w_batch = x_batch.to(device), y_batch.to(device), w_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze()) * w_batch
                loss = loss.mean()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        else:
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        history.append(np.mean(losses))
        # if (epoch) % 10 == 0:
        #     print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    
    return history

def pred_func(
        model: nn.Module,
        x_test: np.ndarray,
        batch_size: int,
        device: torch.device
        ) -> list:
    """
    Predict with the given model.
    Args:
        model: Model to predict
        x_test: Test data
        batch_size: Batch size
        device: Device to run the model
    Returns:
        list: Predicted results
    """
    model = model.to(device)
    model.eval()

    x_test = torch.tensor(x_test, dtype=torch.float)
    test_data = TensorDataset(x_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch[0].to(device)
            y_pred = model(x_batch)
            preds.extend(y_pred.cpu().tolist())
    return preds

def calc_qwk(y_true: list, y_pred: list, prompt_id: int, attribute: str) -> float:
    """
    Calculate the quadratic weighted kappa.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        prompt_id: Prompt ID
        attribute: Attribute name
    Returns:
        float: Quadratic weighted kappa
    """

    minscore, maxscore = get_min_max_scores()[prompt_id][attribute]

    y_true = (maxscore - minscore) * np.array(y_true) + minscore
    y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore).flatten()
    
    return cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[i for i in range(minscore, maxscore+1)])

def get_sample_weight(data_value: np.ndarray, top_p: float, ascending: bool =True):
    """
    Get sample weight for the given data value.
    Args:
        data_value: Data value
        top_p: Top percentage to be selected
        ascending: If True, select the lowest data value
    Returns:
        np.ndarray: Sample weight
    """
    if ascending:
        sorted_data_value = data_value.flatten().argsort()
    else:
        sorted_data_value = data_value.flatten().argsort()[::-1]
    num_elements = int(len(sorted_data_value) * top_p)
    sorted_data_value = sorted_data_value[:num_elements]
    weights = np.ones_like(data_value.flatten())
    for i in sorted_data_value:
        weights[i] = 0
    return weights