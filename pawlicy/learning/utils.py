import os
import inspect
from collections import OrderedDict

import numpy as np
import yaml
from typing import Callable, Union, Tuple, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def lr_schedule(initial_value: Union[float, str], lr_type: str, total_timesteps: Union[int, None] = None) -> Callable[[float], float]:
    """
    Learning rate scheduler that is configured.
    
    Args:
        initial_value: The initial learning rate
        lr_type: The scheduler type
        total_timesteps: The total timesteps to train the agent
    Returns: 
        (function): the scheduler function
    """
    lr_type = lr_type
    timesteps = total_timesteps
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        The new learning rate
        Args:
            progress_remaining: The progress remaining - will decrease from 1 (beginning) to 0
        Returns:
            (float)
        """
        # Cosine Annealing
        if lr_type == "cosine":
            assert timesteps is not None, "Total timesteps required for 'cosine' learning rate scheduler."
            T_max = 1.0 - (total_timesteps * 0.5)/total_timesteps # The maximum progress - currently 10% of the total
            return np.max(0.5 * (1 + np.cos(progress_remaining / T_max * np.pi)) * initial_value, int(1e-5))
        # Linear
        else:
            return np.max(progress_remaining * initial_value, int(1e-5))

    return func

def read_hyperparameters(algorithm: str, verbose=0, custom_hyperparams=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the default hyperparameter config for the given algorithm from
    a common YAML file. These can be overriden using the custom_hyperparams argument.

    Args:
        algorithm: The algorithm for which to get the hyperparameters
        verbose: Whether to print the final hyperparameters in the console or not
        custom_hyperparams: The hyperparameters to change/add
    """
    # Load hyperparameters from yaml file
    file_path = os.path.join(currentdir, "hyperparams.yml")
    with open(file_path) as f:
        hyperparams_dict = yaml.safe_load(f)
        # Find the correct hyperparameters based on the keys
        if algorithm in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[algorithm]
        else:
            raise ValueError(f"Hyperparameters not found for {algorithm}")

    if custom_hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(custom_hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    if verbose > 0:
        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        print(saved_hyperparams)

    return hyperparams, saved_hyperparams

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self._best_reward = 0
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 10000  # log every 10000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        local_env = self.training_env.venv.envs[0]

        # Find the best reward
        reward = np.max(np.array(local_env.episode_returns)) if len(local_env.episode_returns) > 0 else 0
        self._best_reward = self._best_reward if self._best_reward > reward else reward

        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("x_position", local_env.robot.GetBasePosition()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("y_position", local_env.robot.GetBasePosition()[1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("z_position", local_env.robot.GetBasePosition()[2], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("x_velocity", local_env.robot.GetBaseVelocity()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("best_reward", self._best_reward, self.num_timesteps)
            self.tb_formatter.writer.flush()
