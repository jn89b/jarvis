
from pettingzoo.classic import connect_four_v3
import pettingzoo.utils
import os
import glob
from tqdm import tqdm, trange
import yaml
import wandb
import torch
import numpy as np
import unittest
import matplotlib.pyplot as plt
import sys
import matplotlib
import time
import supersuit as ss

from datetime import datetime
from collections import deque
from pettingzoo.test import parallel_api_test
from jarvis.envs import multi_env

import gymnasium
from gymnasium import spaces
from typing import List
from stable_baselines3.common.vec_env import VecNormalize

import pettingzoo
from pettingzoo.test import parallel_api_test, api_test
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

matplotlib.use('TkAgg')
USE_IMPORT = True


# AgileRL stuff
"""
Watch this video 
https://www.youtube.com/results?search_query=rays+rlib+multi+agent+rays rlib
https://medium.com/@jmugan/rllib-for-deep-hierarchical-multiagent-reinforcement-learning-6aa96cdee154
https://github.com/DeUmbraTX/practical_rllib_tutorial

"""

# train with gpu
torch.cuda.set_device(0)

"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
"""


### CUSTOM RLlib TORCH MODEL (Handles Action Masking) ###
class TorchMaskedActions(DQNTorchModel):
    """PyTorch model that supports action masking."""

    def __init__(self, obs_space: spaces.Box, action_space: spaces.Discrete,
                 num_outputs, model_config, name, **kw):
        DQNTorchModel.__init__(self, obs_space, action_space,
                               num_outputs, model_config, name, **kw)

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = spaces.Box(
            shape=(
                obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        """Handles action masking."""
        action_mask = input_dict["obs"]["action_mask"]

        # Compute predicted action logits
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]})

        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e  10, FLOAT_MAX)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        print(self.agent_selection)
        print(self.possible_agents)
        super().step(action)

        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        print(super().observe(self.agent_selection)["action_mask"])
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model = MaskablePPO('MlpPolicy', env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(
        f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


class TestGeneratedData(unittest.TestCase):

    def setUp(self):
        self.test_env = multi_env.raw_env()
        print(isinstance(self.test_env, gymnasium.Env))

    def test_api(self):
        api_test(connect_four_v3.raw_env(),
                 num_cycles=1000, verbose_progress=True)
        # api_test(self.test_env, num_cycles=1000, verbose_progress=True)
        # parallel_api_test(self.test_env, num_cycles=50)

    # def test_train_butterfly_supersuit(self):
    #     self.train_action_mask(env_fn=multi_env, steps=10_000, seed=0)

    def test_env_configuration(self):
        env = multi_env.raw_env()
        # observations, infos = env.reset()

        # print("observations: ", observations)
        # print("infos: ", infos)
        num_steps: int = 20

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # invalid action masking is optional and environment-dependent
                if "action_mask" in info:
                    mask = info["action_mask"]
                elif isinstance(observation, dict) and "action_mask" in observation:
                    mask = observation["action_mask"]
                else:
                    mask = None
                # this is where you would insert your policy
                action = env.action_space(agent).sample(mask)

            env.step(action)

        env.close()
        # for _ in range(num_steps):
        #     actions = {
        #         agent: env.action_spaces[agent].sample() for agent in env.agents}
        #     observations, rewards, dones, _,  infos = env.step(actions)
        #     print("observations", observations)
        #     print("actions", actions)

    def test_train_env(self) -> None:
        # train_action_mask(multi_env, steps=1_000, seed=0)
        ray.init()

        def env_creator():
            env = multi_env.raw_env()

            return env

        env_name = "multi_env"
        register_env(env_name, lambda config: PettingZooEnv(env_creator()))

        test_env = PettingZooEnv(env_creator())
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        print("obs_space: ", obs_space)
        print("act_space: ", act_space)
        ### CONFIGURE RLlib TRAINING ###
        config = (
            DQNConfig()
            .environment(env=env_name)
            .api_stack(enable_rl_module_and_learner=False,
                       enable_env_runner_and_connector_v2=False,)
            .env_runners(num_env_runners=10)
            .training(
                train_batch_size=200,
                hiddens=[],
                dueling=False,
                model={"custom_model": TorchMaskedActions},
            )
            .multi_agent(
                policies={
                    "0", "1"},
                policy_mapping_fn=lambda agent_id, *
                args, **kwargs: str(agent_id),
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
            .debugging(log_level="DEBUG")
            .framework(framework="torch")
            .experimental(_validate_config=False)
        )

        ### TRAINING LOOP ###
        tune.run(
            "DQN",
            name="DQN_Training_RobberPolice",
            stop={"timesteps_total": 10000000 if not os.environ.get(
                "CI") else 50000},
            checkpoint_freq=10,
            config=config.to_dict(),
            storage_path="~/ray_results",
            keep_checkpoints_num=3,  # Retains the last 3 checkpoints
            checkpoint_at_end=True,  # Saves a checkpoint at the end of training
            verbose=3
        )


if __name__ == '__main__':
    unittest.main()
