import matplotlib.pyplot as plt
import copy 
import warnings
from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from jarvis.envs.simple_2d_env import BattleEnv
from jarvis.visualizer.visualizer import Visualizer
from jarvis.assets.Plane2D import Pursuer, Evader 
from jarvis.utils.Vector import StateVector
import numpy as np
import torch
import random
import time 
import pickle as pkl

import gymnasium as gym
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy_2(
    model: "type_aliases.PolicyPredictor",
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

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
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
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

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
    print("observations", observations)
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    x_history = []
    y_history = []
    bad_x = []
    bad_y = []
    overall_x = []
    overall_y = []
    overall_bad_x = []
    overall_bad_y = []
    
    while (episode_counts < episode_count_targets).any():    
        print("observations", observations)
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        
        new_observations, rewards, dones, infos = env.step(actions)
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
                    overall_x.append(copy.deepcopy(x_history))
                    overall_y.append(copy.deepcopy(y_history))
                    overall_bad_x.append(copy.deepcopy(bad_x))
                    overall_bad_y.append(copy.deepcopy(bad_y))
                    x_history = []
                    y_history = []
                    bad_x = []
                    bad_y = []
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

        observations = new_observations
        evader = env.envs[0].agents[0]
        #y = env.envs[0].agents[0].plane.data_handler.y
        x_history.append(copy.deepcopy(evader.state_vector.x))
        y_history.append(copy.deepcopy(evader.state_vector.y))

        pursuer = env.envs[0].battlespace.agents[1]
        bad_x.append(copy.deepcopy(pursuer.state_vector.x))
        bad_y.append(copy.deepcopy(pursuer.state_vector.y))
        
    if render:
        data_vis = Visualizer()
        #loop through overall x
        for i in range(len(overall_x)):
            fig, ax = plt.subplots()
            ax.plot(overall_x[i], overall_y[i], color='b', label='Evader')
            ax.plot(overall_bad_x[i], overall_bad_y[i], color='r', label='Pursuer')
            ax.scatter(overall_x[i][0], overall_y[i][0], color='g', label='Start')
            ax.scatter(overall_bad_x[i][0], overall_bad_y[i][0], color='r', label='Start')
        plt.show()
        env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, env

def create_env():
    return BattleEnv()  # Adjust this to match your environme

if __name__ == '__main__':
    model_name = "PPO_evader_2D_NE"
    vec_normalize_path = "PPO_evader_2D_NE_vecnormalize_2000000.pkl"
    USE_PICKLE_PURSUERS = True
    
    # Load the environment and normalization statistics
    
    num_envs = 1
    # Create a DummyVecEnv for single environment inference
    env = DummyVecEnv([create_env])
    # Set seed number for reproducibility
    seed = 1
    set_global_seed(seed=seed)

    if VecNormalize:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        env.seed(seed=seed)
        
        # Access the mean and standard deviation
        obs_mean = env.obs_rms.mean
        obs_std = np.sqrt(env.obs_rms.var)  # Standard deviation is the square root of the variance

        print("Mean of observations:", obs_mean)
        print("Standard deviation of observations:", obs_std)
    
    # Load the trained model
    model = PPO.load(model_name, env=env, print_system_info=True,
                     device='cuda')
    print("Model loaded.")

    num_times = 0
    num_success = 0
    battle_space_list = []
    reward_list = []
    idx_fail = []

    # Evaluate the policy
    mean_rwd, std_reward,env = evaluate_policy_2(model, 
        model.get_env(), n_eval_episodes=5,
        deterministic=False, render=True)

    # # Visualization
    # data_vis = Visualizer()
    # battlespace = environment.battlespace
    
    # for i, battle_space in enumerate(battle_space_list):
    #     if i in idx_fail:
    #         fig, ax = data_vis.plot_2d_trajectory(battle_space)
    #         #fig, ax =data_vis.plot_attitudes2d(battle_space, ignore_pursuer=True)
    #         #set super title
    #         fig.suptitle(f"LOSE {i}")
    #     else:
    #         fig, ax = data_vis.plot_2d_trajectory(battle_space)
    #         # plot title
    #         #fig, ax =data_vis.plot_attitudes2d(battle_space, ignore_pursuer=True)
    #         fig.suptitle(f"WIN {i}")
    # # Plot the rewards
    # fig, ax = plt.subplots()
    # ax.plot(reward_list)    
    # plt.show()
