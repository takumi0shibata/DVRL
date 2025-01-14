import copy
import logging
import tempfile
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dvrl:
    """
    DVRL class for data valuation using reinforcement learning.
    """

    def __init__(
        self,
        dvrl_data: dict,
        pred_model: nn.Module,
        parameters: dict,
        device: str,
        target_prompt_id: int,
        temp_dir: str = None
    ) -> None:
        """
        Initializes the DVRL model.

        Args:
            dvrl_data (dict): Dictionary containing training and validation data.
            pred_model (nn.Module): Prediction model.
            parameters (dict): Parameters for DVRL.
            device (str): Device to run the model ('cpu' or 'cuda').
            target_prompt_id (int): Prompt ID for the target.
            temp_dir (str, optional): Directory to save temporary files. If None, a temporary directory is created.
        """
        self.x_source = dvrl_data['x_source']
        self.y_source = dvrl_data['y_source'].reshape(-1, 1)
        self.x_dev = dvrl_data['x_dev']
        self.y_dev = dvrl_data['y_dev'].reshape(-1, 1)
        self.x_pseudo = dvrl_data['x_pseudo']
        self.y_pseudo = dvrl_data['y_pseudo'].reshape(-1, 1)
        self.device = device
        self.target_prompt_id = target_prompt_id

        # Network parameters for data value estimator
        self.hidden_dim = parameters.get('hidden_dim', 128)
        self.comb_dim = parameters.get('comb_dim', 64)
        self.outer_iterations = parameters.get('iterations', 1000)
        self.activation_fn = parameters.get('activation', 'relu')
        self.layer_number = parameters.get('layer_number', 2)
        self.inner_iterations = parameters.get('inner_iterations', 100)
        self.batch_size = int(min(parameters.get('batch_size', 32), self.x_source.shape[0]))
        self.learning_rate = parameters.get('learning_rate', 1e-3)
        self.batch_size_predictor = parameters.get('batch_size_predictor', 32)
        self.loss_lambda = parameters.get('loss_lambda', 1.0)
        # self.moving_average = 'moving_average_window' in parameters
        # self.moving_average_window = parameters.get('moving_average_window', 100)

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.data_dim = self.x_source.shape[1]
        self.label_dim = self.y_source.shape[1]

        self.pred_model = pred_model
        self.final_model = copy.deepcopy(pred_model)

        self.use_wandb = parameters.get('wandb', False)

        # Use temporary directory for saving models
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.init_model_path = f'{self.temp_dir}/init_model_{self.target_prompt_id}.pth'

        # Save initial model
        logger.info('Saving the initial model...')
        torch.save(self.pred_model.state_dict(), self.init_model_path)

        # Train baseline model
        self.ori_model = copy.deepcopy(self.pred_model)
        self.ori_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
        logger.info('Training the original model...')
        fit_func(
            self.ori_model,
            self.x_source,
            self.y_source,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device
        )

        # Train validation model
        self.val_model = copy.deepcopy(self.pred_model)
        self.val_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
        logger.info('Training the validation model...')
        fit_func(
            self.val_model,
            self.x_dev,
            self.y_dev,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device
        )

    def train_dvrl(self, metric: str = 'qwk') -> None:
        """
        Trains the DVRL model.

        Args:
            metric (str): Metric to use for the DVRL ('mse', 'qwk', or 'corr').
        """
        # Initialize the data value estimator network
        self.value_estimator = DataValueEstimator(
            input_dim=self.data_dim + self.label_dim,
            hidden_dim=self.hidden_dim,
            comb_dim=self.comb_dim,
            layer_number=self.layer_number,
            activation_fn=self.activation_fn
        ).to(self.device)

        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold).to(self.device)
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)

        # Compute baseline performance
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
            raise ValueError('Unsupported metric. Choose from "mse", "qwk", or "corr".')
        logger.info(f'Baseline Performance ({metric}): {valid_perf:.3f}')

        # Prediction differences
        y_source_valid_pred = pred_func(
            self.val_model,
            self.x_source,
            self.batch_size_predictor,
            self.device
        )
        y_pred_diff = np.abs(self.y_source - y_source_valid_pred)

        # Initialize baseline for reward computation
        baseline = valid_perf

        for iteration in tqdm(range(self.outer_iterations), desc='Training DVRL'):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Batch selection
            batch_indices = np.random.permutation(self.x_source.shape[0])[:self.batch_size]
            x_batch = torch.tensor(self.x_source[batch_indices], dtype=torch.float32).to(self.device)
            y_batch = torch.tensor(self.y_source[batch_indices], dtype=torch.float32).to(self.device)
            y_hat_batch = torch.tensor(y_pred_diff[batch_indices], dtype=torch.float32).to(self.device)

            # Generate selection probabilities
            est_dv_curr = self.value_estimator(x_batch, y_batch, y_hat_batch).squeeze()

            # Sample selection probabilities
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy())
            if np.sum(sel_prob_curr) == 0:
                logger.warning('All zero selection probability. Adjusting selection probabilities.')
                est_dv_curr = torch.full_like(est_dv_curr, 0.5)
                sel_prob_curr = np.random.binomial(1, est_dv_curr.cpu().numpy())

            # Train a new model with the selected data
            new_model = copy.deepcopy(self.pred_model)
            new_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
            fit_func(
                new_model,
                x_batch,
                y_batch,
                self.batch_size_predictor,
                self.inner_iterations,
                self.device,
                sel_prob_curr
            )

            # Validate the new model
            y_valid_hat = pred_func(
                new_model,
                self.x_dev,
                self.batch_size_predictor,
                self.device
            )

            # predict for pseudo
            y_pseudo_hat = pred_func(
                new_model,
                self.x_pseudo,
                self.batch_size_predictor,
                self.device
            )

            # Compute performance metric
            if metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_dev, y_valid_hat)
                pseudo_reward = metrics.mean_squared_error(self.y_pseudo, y_pseudo_hat)
            elif metric == 'qwk':
                dvrl_perf = calc_qwk(self.y_dev, y_valid_hat, self.target_prompt_id, 'score')
                pseudo_reward = calc_qwk(self.y_pseudo, y_pseudo_hat, self.target_prompt_id, 'score')
            elif metric == 'corr':
                dvrl_perf = np.corrcoef(self.y_dev.flatten(), y_valid_hat.flatten())[0, 1]
                pseudo_reward = np.corrcoef(self.y_pseudo.flatten(), y_pseudo_hat.flatten())[0, 1]

            # Compute reward
            reward = dvrl_perf + self.loss_lambda * pseudo_reward - baseline

            # Update the selection network
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
            sel_prob_curr_tensor = torch.tensor(sel_prob_curr, dtype=torch.float32).to(self.device)
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr_tensor, reward_tensor)
            loss.backward()
            dvrl_optimizer.step()

            # # Update the baseline
            # if self.moving_average:
            #     baseline = (
            #         ((self.moving_average_window - 1) * baseline + dvrl_perf) / self.moving_average_window
            #     )

            logger.info(
                f'Iteration: {iteration + 1}, Reward: {reward:.3f}, PSEUDO_Reward: {pseudo_reward:.3f}, DVRL Loss: {loss.item():.3f}, '
                f'Prob MAX: {est_dv_curr.max().item():.3f}, Prob MIN: {est_dv_curr.min().item():.3f}, '
                f'{metric.upper()}: {dvrl_perf:.3f}'
            )

            if self.use_wandb:
                wandb.log(
                    {
                        'Reward': reward,
                        'PSEUDO_Reward': pseudo_reward,
                        'DVRL Loss': loss.item(),
                        'Prob MAX': est_dv_curr.max().item(),
                        'Prob MIN': est_dv_curr.min().item(),
                        metric.upper(): dvrl_perf
                    }
                )

        # Training the final model
        x_source_tensor = torch.tensor(self.x_source, dtype=torch.float32).to(self.device)
        y_source_tensor = torch.tensor(self.y_source, dtype=torch.float32).to(self.device)
        y_pred_diff_tensor = torch.tensor(y_pred_diff, dtype=torch.float32).to(self.device)
        final_data_value = self.value_estimator(x_source_tensor, y_source_tensor, y_pred_diff_tensor).squeeze()
        self.final_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
        fit_func(
            self.final_model,
            self.x_source,
            self.y_source,
            self.batch_size_predictor,
            self.inner_iterations,
            self.device,
            final_data_value.detach().cpu().numpy()
        )

    def dvrl_valuator(self, x_source: np.ndarray, y_source: np.ndarray) -> np.ndarray:
        """
        Estimates the data value for the given source data.

        Args:
            x_source (np.ndarray): Source data features.
            y_source (np.ndarray): Source data labels.

        Returns:
            np.ndarray: Estimated data values.
        """
        x_source_tensor = torch.tensor(x_source, dtype=torch.float32).to(self.device)
        y_source_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32).to(self.device)

        # Calculate prediction difference
        with torch.no_grad():
            output = self.val_model(x_source_tensor)
            y_pred_diff = torch.abs(y_source_tensor - output)

        # Predict data values
        self.value_estimator.eval()
        with torch.no_grad():
            data_value = self.value_estimator(x_source_tensor, y_source_tensor, y_pred_diff).squeeze()

        return data_value.cpu().numpy()

    def dvrl_predict(self, x_target: np.ndarray) -> np.ndarray:
        """
        Predicts the target data using the trained DVRL model.

        Args:
            x_target (np.ndarray): Target data features.

        Returns:
            np.ndarray: Predicted results.
        """
        return pred_func(
            self.final_model,
            x_target,
            self.batch_size_predictor,
            self.device
        )