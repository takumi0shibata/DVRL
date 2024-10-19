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

from dvrl.dvrl_loss import DvrlLossV2
from dvrl.data_value_estimator import DataValueEstimatorV2
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
        self.cluster_assignments = dvrl_data['cluster_assignments']
        self.cluster_centroids = dvrl_data['cluster_centroids']
        self.x_dev = dvrl_data['x_dev']
        self.y_dev = dvrl_data['y_dev'].reshape(-1, 1)
        self.device = device
        self.target_prompt_id = target_prompt_id

        # Network parameters for data value estimator
        self.hidden_dim = parameters.get('hidden_dim', 100)
        self.outer_iterations = parameters.get('iterations', 1000)
        self.activation_fn = parameters.get('activation', nn.Tanh())
        self.layer_number = parameters.get('layer_number', 2)
        self.inner_iterations = parameters.get('inner_iterations', 100)
        self.learning_rate = parameters.get('learning_rate', 1e-3)
        self.batch_size_predictor = parameters.get('batch_size_predictor', 512)

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
        self.ori_model.load_state_dict(torch.load(self.init_model_path))
        logger.info('Training the original model...')
        fit_func(
            self.ori_model,
            self.x_source,
            self.y_source,
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
        self.value_estimator = DataValueEstimatorV2(
            input_dim=self.data_dim + self.label_dim,
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            activation_fn=self.activation_fn
        ).to(self.device)

        dvrl_criterion = DvrlLossV2(self.epsilon, penalty=False).to(self.device)
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

        # Initialize baseline for reward computation
        baseline = valid_perf

        for iteration in range(self.outer_iterations + 1):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Get unique cluster assignments and sort them
            unique_clusters = np.unique(self.cluster_assignments)
            sorted_clusters = np.sort(unique_clusters)

            # Create x_batch with unique cluster centroids
            x_input = [self.cluster_centroids[self.cluster_assignments == cluster][0] for cluster in sorted_clusters]
            x_input = torch.tensor(x_input, dtype=torch.float32).to(self.device)

            # Create y_batch with corresponding labels
            y_input = [self.y_source[self.cluster_assignments == cluster][0] for cluster in sorted_clusters]
            y_input = torch.tensor(y_input, dtype=torch.float32).to(self.device)
            # Generate selection probabilities
            estimated_lambda = self.value_estimator(x_input, y_input).squeeze()
            # logger.info(f'Estimated Lambda: {estimated_lambda}')

            # Create Poisson distribution and sample number of data points per cluster
            poisson_dist = torch.distributions.Poisson(estimated_lambda)
            sampling_data_num = poisson_dist.sample()

            # Select data
            # Initialize lists to store selected data
            x_selected = []
            y_selected = []

            # Iterate through clusters and sample data points
            for cluster_idx, sample_num in enumerate(sampling_data_num):
                sample_num = int(sample_num.item())
                cluster = sorted_clusters[cluster_idx]
                cluster_indices = np.where(self.cluster_assignments == cluster)[0]

                # Ensure sample_num does not exceed cluster size
                sample_num = min(sample_num, len(cluster_indices))

                if sample_num <= 0:
                    continue

                # Randomly sample data points from this cluster
                sampled_indices = np.random.choice(cluster_indices, size=sample_num, replace=False)

                # Add sampled data to x_selected and y_selected
                x_selected.append(self.x_source[sampled_indices])
                y_selected.append(self.y_source[sampled_indices])
            
            if len(x_selected) == 0:
                dvrl_perf = baseline  # No data selected, performance is baseline
                reward = 0
            else:
                # Concatenate selected data
                x_selected = np.concatenate(x_selected, axis=0)
                y_selected = np.concatenate(y_selected, axis=0)
                
                # Convert to torch tensors and move to device
                x_selected = torch.tensor(x_selected, dtype=torch.float32).to(self.device)
                y_selected = torch.tensor(y_selected, dtype=torch.float32).to(self.device)

                # Train a new model with the selected data
                new_model = copy.deepcopy(self.pred_model)
                new_model.load_state_dict(torch.load(self.init_model_path))
                fit_func(
                    new_model,
                    x_selected,
                    y_selected,
                    self.batch_size_predictor,
                    self.inner_iterations,
                    self.device,
                )

                # Validate the new model
                y_valid_hat = pred_func(
                    new_model,
                    self.x_dev,
                    self.batch_size_predictor,
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

            # Ensure reward is a tensor
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)

            # Compute the loss
            loss = dvrl_criterion(sampling_data_num, estimated_lambda, reward_tensor)

            # Backpropagate and update the policy network
            loss.backward()
            dvrl_optimizer.step()

            logger.info(
                f'Iteration: {iteration + 1}, Reward: {reward:.3f}, DVRL Loss: {loss.item():.3f}, '
                f'{metric.upper()}: {dvrl_perf:.3f}, Dataset Size: {len(x_selected)}'
            )

            if self.use_wandb:
                wandb.log(
                    {
                        'Reward': reward,
                        'DVRL Loss': loss.item(),
                        metric.upper(): dvrl_perf
                    }
                )
            
            if iteration == self.outer_iterations:
                # Save the final model
                logger.info('Saving the final model...')
                self.final_model = copy.deepcopy(new_model)

    def dvrl_valuator(self) -> np.ndarray:
        self.value_estimator.eval()
        # Get unique cluster assignments and sort them
        unique_clusters = np.unique(self.cluster_assignments)
        sorted_clusters = np.sort(unique_clusters)

        # Create x_batch with unique cluster centroids
        x_input = [self.cluster_centroids[self.cluster_assignments == cluster][0] for cluster in sorted_clusters]
        x_input = torch.tensor(x_input, dtype=torch.float32).to(self.device)

        # Create y_batch with corresponding labels
        y_input = [self.y_source[self.cluster_assignments == cluster][0] for cluster in sorted_clusters]
        y_input = torch.tensor(y_input, dtype=torch.float32).to(self.device)
        # Generate selection probabilities
        estimated_lambda = self.value_estimator(x_input, y_input).squeeze()

        return estimated_lambda.detach().cpu().numpy()
    
    def predict(self, x_test, y_test):
        y_test_hat = pred_func(
            self.final_model,
            x_test,
            self.batch_size_predictor,
            self.device
        )
        qwk = calc_qwk(y_test, y_test_hat, self.target_prompt_id, 'score')
        return qwk
    
    def predict_baseline(self, x_test, y_test):
        y_test_hat = pred_func(
            self.ori_model,
            x_test,
            self.batch_size_predictor,
            self.device
        )
        qwk = calc_qwk(y_test, y_test_hat, self.target_prompt_id, 'score')
        return qwk