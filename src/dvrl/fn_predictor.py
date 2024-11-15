"""Utility functions for DVRL model."""

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from utils.general_utils import get_min_max_scores
from typing import Union, List, Dict
import copy

def calc_qwk(y_true: list, y_pred: list, prompt_id: int, attribute: str, weights='quadratic') -> float:
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

    y_true = np.round((maxscore - minscore) * np.array(y_true) + minscore).flatten()
    y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore).flatten()
    
    return cohen_kappa_score(y_true, y_pred, weights=weights, labels=[i for i in range(minscore, maxscore+1)])


def fit_func(
    model: nn.Module,
    x_train: list | np.ndarray,
    y_train: np.ndarray,
    optimizer_fn: optim.Optimizer,
    lr: float,
    batch_size: int,
    iterations: int,
    device: str,
    prompt_id: int,
    metric: str = 'qwk',  # Metric for dev set evaluation ('qwk', 'mse', or 'corr')
    x_dev: list | np.ndarray = None,
    y_dev: np.ndarray = None,
    use_final_epoch_model: bool = False  # Flag to use final epoch model instead of best dev model
) -> nn.Module:
    """
    Fits the model to the training data while validating on the dev set.

    Args:
        model (nn.Module): The model to fit.
        x_train (list or np.ndarray): List of training data features if multiple inputs are needed; single np.ndarray otherwise.
        y_train (np.ndarray): Training data labels.
        optimizer_fn (optim.Optimizer): Optimizer function to use.
        lr (float): Learning rate for training.
        batch_size (int): Batch size for training.
        iterations (int): Number of epochs to train.
        device (str): Device to use ('cpu' or 'cuda').
        prompt_id (int): Prompt ID.
        metric (str): Metric for dev set evaluation ('qwk', 'mse', or 'corr').
        x_dev (list or np.ndarray): List of dev data features if multiple inputs are needed; single np.ndarray otherwise.
        y_dev (np.ndarray): Dev data labels.
        use_final_epoch_model (bool): Flag to use final epoch model instead of best dev model.

    Returns:
        nn.Module: The best or final epoch model based on `use_final_epoch_model`.
    """
    optimizer = optimizer_fn(model.parameters(), lr=lr)  # Adjust learning rate as needed
    criterion = nn.MSELoss()  # Adjust loss function as necessary, e.g., CrossEntropyLoss for classification
    model.to(device)

    best_model = copy.deepcopy(model)
    if metric == 'mse':
        best_dev_score = float('inf')
    elif metric in ['qwk', 'corr']:
        best_dev_score = float('-inf')

    # Prepare DataLoader for training data
    if isinstance(x_train, list):
        x_train_tensors = [torch.tensor(x_train[0], dtype=torch.long)] + [torch.tensor(x, dtype=torch.float32) for x in x_train[1:]]
    else:
        x_train_tensors = [torch.tensor(x_train, dtype=torch.float32)]
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(*x_train_tensors, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Convert dev data to tensors for evaluation if needed
    if not use_final_epoch_model and x_dev is not None and y_dev is not None:
        if isinstance(x_dev, list):
            x_dev_tensors = [torch.tensor(x_dev[0], dtype=torch.long).to(device)] + [torch.tensor(x, dtype=torch.float32).to(device) for x in x_dev[1:]]
        else:
            x_dev_tensors = [torch.tensor(x_dev, dtype=torch.float32).to(device)]
        y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32).to(device)

    for epoch in range(iterations):
        model.train()

        # Iterate over batches
        for batch in train_loader:
            batch_x = [b.to(device) for b in batch[:-1]]  # All inputs except the last are x
            batch_y = batch[-1].to(device)  # Last item in batch is y

            optimizer.zero_grad()
            outputs = model(*batch_x)  # Pass multiple inputs as unpacked arguments
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on dev set if using best dev model selection
        if not use_final_epoch_model and x_dev is not None and y_dev is not None:
            model.eval()
            with torch.no_grad():
                dev_outputs = model(*x_dev_tensors)
                if metric == 'mse':
                    dev_score = mean_squared_error(y_dev_tensor.cpu().numpy(), dev_outputs.cpu().numpy())
                    is_better = dev_score < best_dev_score  # Lower MSE is better
                elif metric == 'qwk':
                    dev_score = calc_qwk(y_dev_tensor.cpu().numpy(), dev_outputs.cpu().numpy(), prompt_id, 'score')
                    is_better = dev_score > best_dev_score  # Higher QWK is better
                elif metric == 'corr':
                    dev_score = np.corrcoef(y_dev_tensor.cpu().numpy().flatten(), dev_outputs.cpu().numpy().flatten())[0, 1]
                    is_better = dev_score > best_dev_score  # Higher correlation is better

                # Update best model if current dev performance is better
                if is_better:
                    best_dev_score = dev_score
                    best_model = copy.deepcopy(model)

    # Decide which model to return
    return model if use_final_epoch_model else best_model

def pred_func(
    model: nn.Module,
    x_test: list | np.ndarray,
    batch_size: int,
    device: str
) -> np.ndarray:
    """
    Predicts the labels for the test data using the trained model.

    Args:
        model (nn.Module): The trained model.
        x_test (list or np.ndarray): List of test data features if multiple inputs are needed; single np.ndarray otherwise.
        batch_size (int): Batch size for prediction.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        np.ndarray: Predicted labels for the test data.
    """
    model.eval()
    if isinstance(x_test, list):
        x_test_tensors = [torch.tensor(x_test[0], dtype=torch.long).to(device)] + [torch.tensor(x, dtype=torch.float32).to(device) for x in x_test[1:]]
    else:  # Assuming x_test is a single numpy array or tensor if not a list
        x_test_tensors = [torch.tensor(x_test, dtype=torch.float).to(device)]
    test_dataset = TensorDataset(*x_test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch_x = [b.to(device) for b in batch]
            outputs = model(*batch_x)
            preds.extend(outputs.cpu().tolist())

    return np.array(preds)
