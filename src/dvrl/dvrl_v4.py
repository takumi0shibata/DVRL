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
from dvrl.fn_predictor import fit_func, pred_func, calc_qwk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dvrl:
    """
    DVRL class for data valuation using reinforcement learning.
    """

    def __init__(
        self,
        train_data: dict,
        x_source: np.ndarray,
        y_source: np.ndarray,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        pred_model: nn.Module,
        parameters: dict,
        device: str,
        target_prompt_id: int,
        predictor_config: dict,
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
        self.train_data = train_data
        self.x_source = x_source
        self.y_source = y_source.reshape(-1, 1)
        self.x_dev = x_dev
        self.y_dev = y_dev.reshape(-1, 1)
        self.device = device
        self.target_prompt_id = target_prompt_id
        self.predictor_config = predictor_config

        # Network parameters for data value estimator
        self.hidden_dim = parameters.get('hidden_dim', 100)
        self.comb_dim = parameters.get('comb_dim', 64)
        self.outer_iterations = parameters.get('iterations', 1000)
        self.activation_fn = parameters.get('activation', 'relu')
        self.layer_number = parameters.get('layer_number', 2)
        self.lr = parameters.get('lr', 0.001)

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
        self.ori_model = fit_func(
            self.ori_model,
            self.x_source,
            self.y_source,
            self.predictor_config.optimizer,
            self.predictor_config.lr,
            self.predictor_config.batch_size,
            self.predictor_config.epochs,
            self.device,
            self.target_prompt_id,
            'qwk',
            self.x_dev,
            self.y_dev,
            self.predictor_config.use_final_epoch_model
        )

        # Train validation model
        self.val_model = copy.deepcopy(self.pred_model)
        self.val_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
        logger.info('Training the validation model...')
        self.val_model = fit_func(
            self.val_model,
            self.x_dev,
            self.y_dev,
            self.predictor_config.optimizer,
            self.predictor_config.lr,
            self.predictor_config.batch_size,
            self.predictor_config.epochs,
            self.device,
            self.target_prompt_id,
            'mse',
            self.x_source,
            self.y_source,
            self.predictor_config.use_final_epoch_model
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
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.lr)

        # Compute baseline performance
        y_valid_hat = pred_func(
            self.ori_model,
            self.x_dev,
            self.predictor_config.batch_size,
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
            self.predictor_config.batch_size,
            self.device
        )
        y_pred_diff = np.abs(self.y_source - y_source_valid_pred)
        # Calculate mean prediction difference per cluster
        cluster_mean_pred_diff = {}
        unique_clusters = np.unique(self.train_data['cluster'])
        # Calculate mean prediction difference for each cluster
        for c in unique_clusters:
            cluster_mask = self.train_data['cluster'] == c
            cluster_mean_pred_diff[c] = np.mean(y_pred_diff[cluster_mask])

        # Initialize baseline for reward computation
        baseline = valid_perf

        # Prepare data for DVRL
        cluster = np.unique(self.train_data['cluster'])
        x_cluster_centroids = np.array([np.mean(self.train_data['embedding'][self.train_data['cluster'] == c], axis=0) for c in cluster])
        x_cluster_centroids = torch.tensor(x_cluster_centroids, dtype=torch.float32).to(self.device)
        y_cluster = np.array([np.mean(self.train_data['scaled_score'][self.train_data['cluster'] == c], axis=0) for c in cluster])
        y_cluster = torch.tensor(y_cluster.reshape(-1, 1), dtype=torch.float32).to(self.device)

        # Map the mean prediction differences back to each data point
        y_pred_diff_by_cluster = np.array([cluster_mean_pred_diff.get(c, 1) for c in cluster]).reshape(-1, 1)
        y_pred_diff_by_cluster = torch.tensor(y_pred_diff_by_cluster, dtype=torch.float32).to(self.device)

        # Train DVRL
        for iteration in tqdm(range(self.outer_iterations), desc='Training DVRL'):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Generate selection probabilities
            est_dv_curr = self.value_estimator(x_cluster_centroids, y_cluster, y_pred_diff_by_cluster).squeeze()

            # Sample selection probabilities
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy())
            if np.sum(sel_prob_curr) == 0:
                logger.warning('All zero selection probability. Adjusting selection probabilities.')
                est_dv_curr = torch.full_like(est_dv_curr, 0.5)
                sel_prob_curr = np.random.binomial(1, est_dv_curr.cpu().numpy())

            # Train a new model with the selected data
            new_model = copy.deepcopy(self.pred_model)
            new_model.load_state_dict(torch.load(self.init_model_path, weights_only=True))
            for_using_cluster = cluster[sel_prob_curr == 1]
            cluster_mask = np.isin(self.train_data['cluster'], for_using_cluster)
            new_model = fit_func(
                new_model,
                self.x_source[cluster_mask],
                self.y_source[cluster_mask],
                self.predictor_config.optimizer,
                self.predictor_config.lr,
                self.predictor_config.batch_size,
                self.predictor_config.epochs,
                self.device,
                self.target_prompt_id,
                metric,
                self.x_dev,
                self.y_dev,
                self.predictor_config.use_final_epoch_model
            )

            # Validate the new model
            y_valid_hat = pred_func(
                new_model,
                self.x_dev,
                self.predictor_config.batch_size,
                self.device
            )

            # Compute performance metric
            if metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_dev, y_valid_hat)
                reward = baseline - dvrl_perf
            elif metric == 'qwk':
                dvrl_perf = calc_qwk(self.y_dev, y_valid_hat, self.target_prompt_id, 'score')
                reward = dvrl_perf - baseline
            elif metric == 'corr':
                dvrl_perf = np.corrcoef(self.y_dev.flatten(), y_valid_hat.flatten())[0, 1]
                reward = dvrl_perf - baseline

            # Update the selection network
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
            sel_prob_curr_tensor = torch.tensor(sel_prob_curr, dtype=torch.float32).to(self.device)
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr_tensor, reward_tensor)
            loss.backward()
            dvrl_optimizer.step()

            if iteration % 1 == 0:
                logger.info(f'Iteration: {iteration + 1}, Reward: {reward:.3f}, Loss: {loss.item():.3f}, {metric.upper()}: {dvrl_perf:.3f}, ValueMean: {np.round(est_dv_curr.mean().detach().cpu().numpy(), 2)}, ValueStd: {np.round(est_dv_curr.std().detach().cpu().numpy(), 2)}')
                logger.info(f'Values: {est_dv_curr.detach().cpu().numpy()}')
                
            if self.use_wandb:
                wandb.log(
                    {
                        'Reward': reward,
                        'Loss': loss.item(),
                        metric.upper(): dvrl_perf,
                        'ValueMean': est_dv_curr.mean().detach().cpu().numpy(),
                        'ValueStd': est_dv_curr.std().detach().cpu().numpy()
                    }
                )

        self.value_estimator.eval()
        # Generate selection probabilities
        est_dv_curr = self.value_estimator(x_cluster_centroids, y_cluster, y_pred_diff_by_cluster).squeeze()

        return {
            'cluster': cluster,
            'values': est_dv_curr.detach().cpu().numpy()
        }