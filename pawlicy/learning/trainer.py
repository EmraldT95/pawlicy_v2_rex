import os
import inspect
from xml.dom import NotFoundErr

import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList

from pawlicy.envs.wrappers import NormalizeActionWrapper
from pawlicy.learning import utils

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
SAVE_DIR = os.path.join(currentdir, "agents")

ALGORITHMS = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "DDPG": DDPG}

class Trainer:
    """
    The trainer class provides some basic methods to train an agent using different algorithms
    available in stable_baselines3

    Args:
        env: The gym environment to train on.
        eval_env: The environment to evaluate on
        algorithm: The algorithm to use.
        max_episode_steps: The no. of steps per episode
    """
    def __init__(self, env, eval_env=None, algorithm="SAC", max_episode_steps=100, save_path=None):
        self._eval_env = eval_env
        self._algorithm = algorithm
        self._max_episode_steps = max_episode_steps
        self._env = self.setup_env(env, self._max_episode_steps)
        self._save_path = save_path
        # Setup the evaluation environment as well, if available
        if eval_env is not None:
            self._eval_env = self.setup_env(eval_env, self._max_episode_steps // 2)

    def train(self, override_hyperparams={}):
        """
        Trains an agent to use the environment to maximise the rewards while performing
        a specific task. This will tried out with multiple other algorithms later for
        benchmarking purposes.

        Args:
            override_hyperparams: The hyperparameters to override/add to the default config
        """
        # Get the default hyperparameters and override if needed
        _, hyperparameters = utils.read_hyperparameters(self._algorithm, 1, override_hyperparams)

        # Sanity checks
        n_timesteps = hyperparameters.pop("n_timesteps", None)
        if n_timesteps is None:
            raise ValueError("The hyperparameter 'n_timesteps' is missing.")
        eval_frequency = hyperparameters.pop("eval_freq", 5000)
        scheduler_type = hyperparameters.pop("learning_rate_scheduler", None)
        lr = hyperparameters.pop("learning_rate", float(1e-3))
        noise_type = hyperparameters.pop("noise_type", "normal")
        noise_std = hyperparameters.pop("noise_std", 0.0)

        # The noise objects for TD3
        if self._algorithm == "TD3" or self._algorithm == "DDPG":
            policy_kwargs = dict(net_arch=[400, 300])
            if noise_type == "normal":
                action_noise = NormalActionNoise(mean=np.zeros(12), sigma=noise_std * np.ones(12))
            # Add these to the hyperparameters
            hyperparameters.update({"action_noise": action_noise, "policy_kwargs": policy_kwargs})

        # Setup up learning rate scheduler arguments, if needed
        if scheduler_type is not None:
            lr_scheduler_args = {
                "lr_type": scheduler_type,
                "total_timesteps": n_timesteps
            }

        # Create all the needed directories
        log_dir, eval_dir = self._create_directories(self._save_path)

        # Use the appropriate algorithm
        self._model = ALGORITHMS[self._algorithm](env=self._env,
                                                    verbose=1,
                                                    learning_rate=utils.lr_schedule(lr, **lr_scheduler_args) if scheduler_type is not None else lr,
                                                    tensorboard_log=log_dir,
                                                    **hyperparameters)

        # Train the model (check if evaluation is needed)
        if self._eval_env is not None:
            eval_callback = utils.Eval_Callback(eval_env=self._eval_env,
                                        best_model_save_path=eval_dir,
                                        log_path=eval_dir,
                                        eval_freq=eval_frequency,
                                        deterministic=True,
                                        render=False,
                                        verbose=1)
                                        # This is needed to generate the video after evaluation is complete
                                        # callback_after_eval=utils.After_Eval_Callback(1, eval_dir))

            callbacks = CallbackList([eval_callback, utils.TensorboardCallback()])

            self._model.learn(n_timesteps,
                                log_interval=1000,
                                reset_num_timesteps=False,
                                callback=callbacks)
        else:
            self._model.learn(n_timesteps,
                                log_interval=1000,
                                reset_num_timesteps=False,
                                callback=utils.TensorboardCallback())

        # Return the trained model
        return self._model

    def _create_directories(self, save_path=None):
        """Creates all the directories to save the logs and the models
        
        Args:
            save_path: The path in which all the directories will be made in.

        Returns:
            The tensorboard and evaluation log direcrtory paths
        """

        # If no explicit path is mentioned, we create one based the algorithm used to train
        if save_path is None:
            self._save_path = os.path.join(SAVE_DIR, f"{self._algorithm}")
        # Create the directory, if it doesn't exist
        os.makedirs(self._save_path, exist_ok=True)

        # Where to store the tensorboard logs
        log_dir = os.path.join(self._save_path, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        eval_dir = os.path.join(self._save_path, "eval")
        if self._eval_env:
            os.makedirs(eval_dir, exist_ok=True)

        return log_dir, eval_dir

    def save_model(self, save_path=None, save_replay_buffer=False):
        """Saves the trained model. Also saves the replay buffer

        Args:
            save_path: The path to save the model in
            save_replay_buffer: Whether to save the replay buffer or not
        """

        if save_path is None:
            save_path = self._save_path
        else:
            os.makedirs(save_path, exist_ok=True)

        # Save the model
        self._model.save(os.path.join(save_path, self._algorithm))
        # Save the replay buffer, only if needed
        if save_replay_buffer:
            self._model.save_replay_buffer(os.path.join(save_path, f"{self._algorithm}_replay_buffer"))

        print(f"Model saved in path: {save_path}")

    def load_model(self, load_path=None):
        """Loads a trained model from the given path. Also loads
        the replay buffer, if available.

        Args:
            model_path: path to the directory containing the model/replay_buffer
        """
        # Load the best eval model, if the file exists
        model_path = os.path.join(load_path, "eval/best_model.zip")
        if os.path.exists(model_path):
            model = ALGORITHMS[self._algorithm].load(model_path)
        else:
            raise NotFoundErr(
                "The model could not be found in the given directory. Please ensure the file is name as 'best_model.zip'.")

        # Load the replay buffer, if the file exists
        replay_buffer_path = os.path.join(load_path, f"{self._algorithm}_replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        
        return model

    def test(self, model_path=None):
        """Tests the agent

        Args:
            model_path: The path to the directory containing the model file
        """
        # If no explicit path is mentioned, we create one based the algorithm used to train
        if model_path is not None:
            self._model = self.load_model(model_path)
        elif self._save_path is not None:
            self._model = self.load_model(self._save_path)
        else:
            raise ValueError("A path must be provided to load the model.")

        obs = self._env.reset()
        rewards = []
        ep_reward = 0
        for _ in range(5000):
            action, _states = self._model.predict(obs, deterministic=True)
            obs, reward, done, info = self._env.step(action)
            ep_reward += reward
            if done:
                rewards.append(ep_reward)
                obs = self._env.reset()
        print("Mean reward: ", np.mean(np.array(rewards)))

    def setup_env(self, env, max_episode_steps):
        """Modifies the environment to suit to the needs of stable_baselines3.

        Args:
            max_episode_steps: The number of steps per episode
        """
        # Normalize the action space
        env = NormalizeActionWrapper(env)
        # Set the number of steps for each episode
        env = TimeLimit(env, max_episode_steps)
        # To monitor training stats
        env = Monitor(env)
        # a simple vectorized wrapper
        env = DummyVecEnv([lambda: env])
        # Normalizes the observation space and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        return env;

    @property
    def model(self):
        return self._model

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value
