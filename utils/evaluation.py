from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error
from utils.general_utils import get_min_max_scores

# 訓練関数の定義
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()

    losses = []
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)
    for x_train, y_train, linguistic_train, readability_train, _ in progress_bar:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        linguistic_train = linguistic_train.to(device)
        readability_train = readability_train.to(device)

        # predict
        y_pred = model(x_train, linguistic_train, readability_train)

        # calculate loss
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
def evaluate_model(model, data_loader, loss_fn, device, attribute):
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