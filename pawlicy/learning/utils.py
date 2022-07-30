import os
import subprocess
import glob
import inspect
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import pybullet as p
import numpy as np
import yaml
import matplotlib.pyplot as plt
import gym
from typing import Callable, Union, Tuple, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization

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

class Eval_Callback(EvalCallback):
    """Callback function used in the evaluation process"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "callback_after_eval" in kwargs.keys():
            self._save_video = True
            os.makedirs(os.path.join(self.best_model_save_path, "videos"), exist_ok=True)
        else:
            self._save_video = False

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
        # Saving the image plots if the evaluation is called at every 10000th step of training
        if self._save_video and self.num_timesteps % 10000 == 0:
            # Save every 5th image
            if self._save_plot_iteration % 5 == 0:
                plt.imshow(self.eval_env.render())
                plt.savefig(os.path.join(self.best_model_save_path, f'videos/plot_{self._plot_iteration}.png'))
                self._plot_iteration += 1
            self._save_plot_iteration += 1

    def _on_step(self) -> bool:
        self._plot_iteration = 0
        self._save_plot_iteration = 0
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class After_Eval_Callback(BaseCallback):
    """Call back function called after the evaluation is complete"""
    
    def __init__(self, verbose: int = 0, best_model_save_path = None):
        self.best_model_save_path = best_model_save_path
        super().__init__(verbose)

    def _on_step(self) -> bool:
        super()._on_step()
        if int(self.num_timesteps % 10000) == 0:
            self.generate_video()

    def generate_video(self):
        """Creates a video using the plots"""
        os.chdir(os.path.join(self.best_model_save_path, "videos"))
        subprocess.call([
            'ffmpeg', '-i', 'plot_%d.png', '-r', '30', f'model_{self.num_timesteps}.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
