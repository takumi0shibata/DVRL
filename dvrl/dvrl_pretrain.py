# coding=utf-8
"""
DVRL pretraining the prediction model script.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def pretrain(predictor_model, data_loader, inner_learning_rate, device, epoch=50):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(predictor_model.parameters(), lr=inner_learning_rate)

    predictor_model = predictor_model.to(device)
    predictor_model.train()
    for i in range(epoch):
        print(f"Epoch {i + 1}/{epoch}")
        losses = []
        progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)
        for x_train, y_train, linguistic_train, readability_train, _ in progress_bar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            linguistic_train = linguistic_train.to(device)
            readability_train = readability_train.to(device)

            # predict
            y_pred = predictor_model(x_train, linguistic_train, readability_train)

            # calculate loss
            loss = loss_fn(y_pred.squeeze(), y_train.squeeze())
            losses.append(loss.item())
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

    return predictor_model
