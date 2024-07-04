
"""DVRL class for data valuation using reinforcement learning"""

import copy
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import wandb

from dvrl.dvrl_loss import DvrlLoss
from dvrl.data_value_estimator import DataValueEstimator
from utils.dvrl_utils import fit_func, pred_func, calc_qwk


class Dvrl(object):

    def __init__(
        self,
        dvrl_data: dict,
        pred_model: nn.Module,
        parameters: dict,
        device: str,
        target_prompt_id: int
    ) -> None:
        """
        Args:
            x_source: Training data
            y_source: Training labels
            x_dev: Validation data
            y_dev: Validation labels
            pred_model: Prediction model
            parameters: Parameters for DVRL
            device: Device to run the model
            target_prompt_id: Prompt id for the target
        """

        self.x_source = dvrl_data['x_source']
        self.y_source = dvrl_data['y_source'].reshape(-1, 1)
        self.x_dev = dvrl_data['x_dev']
        self.y_dev = dvrl_data['y_dev'].reshape(-1, 1)
        self.device = device
        self.target_prompt_id = target_prompt_id

        # Network parameters for data value estimator
        self.hidden_dim = parameters['hidden_dim']
        self.comb_dim = parameters['comb_dim']
        self.outter_iterations = parameters['iterations']
        self.act_fn = parameters['activation']
        self.layer_number = parameters['layer_number']
        self.inner_iterations = parameters['inner_iterations']
        self.batch_size = int(np.min([parameters['batch_size'], self.x_source.shape[0]]))
        self.learning_rate = parameters['learning_rate']
        self.batch_size_predictor = parameters['batch_size_predictor']
        if 'moving_average_window' in parameters:
            self.moving_average = True
            self.moving_average_window = parameters['moving_average_window']
        else:
            self.moving_average = False

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.data_dim = self.x_source.shape[1]
        self.label_dim = self.y_source.shape[1]

        self.pred_model = pred_model
        self.final_model = pred_model

        self.wandb = parameters['wandb']

        # save initial model
        print('Saving the initial model...')
        os.makedirs('tmp', exist_ok=True)
        torch.save(self.pred_model.state_dict(), f'tmp/init_model{self.target_prompt_id}.pth')

        # train baseline model
        self.ori_model = copy.deepcopy(self.pred_model)
        self.ori_model.load_state_dict(torch.load(f'tmp/init_model{self.target_prompt_id}.pth'))
        print('Training the original model...')
        fit_func(
            self.ori_model,
            self.x_source,
            self.y_source,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device
        )

        self.val_model = copy.deepcopy(self.pred_model)
        self.val_model.load_state_dict(torch.load(f'tmp/init_model{self.target_prompt_id}.pth'))
        print('Training the validation model...')
        fit_func(
            self.val_model,
            self.x_dev,
            self.y_dev,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device
        )


    def train_dvrl(
        self,
        metric: str = 'qwk',
    ) -> None:
        """
        Train the DVRL model
        Args:
            metric: Metric to use for the DVRL
                mse or qwk or corr
        """
        # selection network
        self.value_estimator = DataValueEstimator(
            self.data_dim+self.label_dim,
            self.hidden_dim,
            self.comb_dim,
            self.layer_number,
            self.act_fn
        ).to(self.device)
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold).to(self.device)
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)

        # baseline performance
        y_valid_hat = pred_func(
            self.ori_model,
            self.x_dev,
            self.batch_size_predictor,
            self.device
        )
        if metric == 'mse':
            valid_perf = metrics.mean_squared_error(self.y_dev, y_valid_hat)
        elif metric == 'qwk':
            valid_perf = calc_qwk(self.y_dev, y_valid_hat, self.target_prompt_id, 'score')
        elif metric == 'corr':
            valid_perf = np.corrcoef(self.y_dev.flatten(), y_valid_hat.flatten())[0, 1]
        else:
            raise ValueError('Metric not supported')
        print(f'Baseline Performance {metric}: {valid_perf:.3f}')

        # Prediction differences
        y_source_valid_pred = pred_func(
            self.val_model,
            self.x_source,
            self.batch_size_predictor,
            self.device
        )
        y_pred_diff = np.abs(self.y_source - y_source_valid_pred)

        if self.moving_average:
            baseline = 0
        else:
            baseline = valid_perf
        
        for iter in tqdm(range(self.outter_iterations)):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Batch selection
            batch_idx = np.random.permutation(self.x_source.shape[0])[:self.batch_size]

            x_batch = torch.tensor(self.x_source[batch_idx], dtype=torch.float).to(self.device)
            y_batch = torch.tensor(self.y_source[batch_idx], dtype=torch.float).to(self.device)
            y_hat_batch = torch.tensor(y_pred_diff[batch_idx], dtype=torch.float).to(self.device)

            # Generates the selection probability
            est_dv_curr = self.value_estimator(x_batch, y_batch, y_hat_batch).squeeze()

            # Samples the selection probability
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy(), est_dv_curr.shape)
            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                print('All zero selection probability')
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

            new_model = self.pred_model
            new_model.load_state_dict(torch.load(f'tmp/init_model{self.target_prompt_id}.pth'))
            fit_func(
                new_model,
                x_batch,
                y_batch,
                self.batch_size_predictor,
                self.inner_iterations,
                self.device,
                sel_prob_curr,
            )
            y_valid_hat = pred_func(
                new_model,
                self.x_dev,
                self.batch_size_predictor,
                self.device
            )

            # reward computation
            if metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_dev, y_valid_hat)
            elif metric == 'qwk':
                dvrl_perf = calc_qwk(self.y_dev, y_valid_hat, self.target_prompt_id, 'score')
            elif metric == 'corr':
                dvrl_perf = np.corrcoef(self.y_dev.flatten(), y_valid_hat.flatten())[0, 1]
            reward = dvrl_perf - baseline

            # update the selection network
            reward = torch.tensor([reward]).to(self.device)
            sel_prob_curr = torch.tensor(sel_prob_curr, dtype=torch.float).to(self.device)
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr, reward)
            loss.backward()
            dvrl_optimizer.step()

            # update the baseline
            if self.moving_average:
                baseline = ((self.moving_average_window - 1) / self.moving_average_window) * baseline + (dvrl_perf / self.moving_average_window)

            print(f'Iteration: {iter+1}, Reward: {reward.item():.3f}, DVRL Loss: {loss.item():.3f}, Prob MAX: {torch.max(est_dv_curr).item():.3f}, Prob MIN: {torch.min(est_dv_curr).item():.3f}, {metric}: {dvrl_perf:.3f}')

            if self.wandb:
                wandb.log(
                    {
                        'Reward': reward.item(),
                        'DVRL Loss': loss.item(),
                        'Prob MAX': torch.max(est_dv_curr).item(),
                        'Prob MIN': torch.min(est_dv_curr).item(),
                        metric: dvrl_perf
                    }
                )


        # Training the final model
        x_source = torch.tensor(self.x_source, dtype=torch.float).to(self.device)
        y_source = torch.tensor(self.y_source, dtype=torch.float).to(self.device)
        y_pred_diff = torch.tensor(y_pred_diff, dtype=torch.float).to(self.device)
        final_data_value = self.value_estimator(x_source, y_source, y_pred_diff).squeeze()
        self.final_model.load_state_dict(torch.load(f'tmp/init_model{self.target_prompt_id}.pth'))
        fit_func(
            self.final_model,
            self.x_source,
            self.y_source,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device,
            final_data_value
        )


    def dvrl_valuator(self, x_source: np.ndarray, y_source: np.ndarray) -> np.ndarray:
        """
        Estimate the given data value.
        Args:
            x_source: Training data
            y_source: Training labels
        Returns:
            data_value: Estimated data value
        """
        x_source = torch.tensor(x_source, dtype=torch.float).to(self.device)
        y_source = torch.tensor(y_source, dtype=torch.float).view(-1, 1).to(self.device)

        # first calculate the prection difference
        output = self.val_model(x_source)
        y_pred_diff = torch.abs(y_source - output)

        # predict the value
        data_value = self.value_estimator(x_source, y_source, y_pred_diff)
        
        return data_value.cpu().detach().numpy()
    

    def dvrl_predict(self, x_target: np.ndarray) -> np.ndarray:
        """
        Predict the given data using the DVRL model
        Args:
            x_target: target data
        Returns:
            target_results: Predicted results
        """

        test_results = pred_func(
            self.final_model,
            x_target,
            self.batch_size_predictor,
            self.device
        )

        return test_results