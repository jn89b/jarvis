from __future__ import annotations
from typing import List, Dict, Tuple

from pettingzoo import ParallelEnv
from copy import copy
import random

import os

import gymnasium
import numpy as np
import functools
import pygame
from gymnasium import spaces
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector


MIN_IDX_BOARD: int = 0
MAX_IDX_BOARD: int = 6

"""
Robber and Police game
Where agents are placed on a 7x7 grid. The robber must reach the escape cell (randomly placed) without being caught by the police.
The police must catch the robber before they reach the escape cell.
Agents: [prisoner, guard] -> [0, 1]
Observation: (4, 2) -> (xy, robber_or_police)
    : [
        [current_x, current_y, opp_rel_x, opp_rel_y]
    ]


This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import rps_v2` |
|--------------------|-----------------------------------------|
| Actions            | Discrete                                |
| Parallel API       | Yes                                     |
| Manual Control     | No                                      |
| Agents             | `agents= ['player_0', 'player_1']`      |
| Agents             | 2                                       |
| Action Shape       | Discrete(2)                             |
| Action Values      | Discrete(3)                             |
| Observation Shape  | Discrete(4)                             |
| Observation Values | Discrete(4)                             |

ACTION = 0: Left
ACTION = 1: Right
ACTION = 2: Down
ACTION = 3: Up

"""


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "test",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None,):
        EzPickle.__init__(self)
        super().__init__()

        self.prisoner_x = 0
        self.prisoner_y = 0
        self.guard_x = MAX_IDX_BOARD
        self.guard_y = MAX_IDX_BOARD
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)
        self.timestep = 0
        self.agent_name_mapping: Dict[str, int] = {
            "prisoner": 0,
            "guard":  1}
        self.agents = list(self.agent_name_mapping.values())
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.terminations: Dict[int, bool] = {a: False for a in self.agents}
        self.truncations: Dict[int, bool] = {a: False for a in self.agents}
        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up
        self.observation_spaces = {}
        for i, agent in enumerate(self.agents):

            low_obs: List[int] = [0, 0, -6, -6]
            high_obs: List[int] = [6, 6, 6, 6]
            obs_box: spaces.Box = spaces.Box(low=np.array(low_obs),
                                             high=np.array(high_obs),
                                             dtype=np.float32)
            self.observation_spaces[i] = spaces.Dict({
                "observation": obs_box,
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.int8)
            })
        self.action_spaces = {agent: Discrete(4) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> Dict[str, spaces.Box]:
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # current x, current y, relative x, relative y
        return self.observation_spaces[agent]
    #     # return MultiDiscrete([7 * 7 - 1] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        return self.action_spaces[agent]

    def _get_action_mask(self, agent: int) -> Tuple(int, np.ndarray):
        action_mask = np.ones(4, dtype=np.int8)
        if agent == self.agent_name_mapping["prisoner"]:
            # Generate action masks
            action_mask = np.ones(4, dtype=np.int8)
            if self.prisoner_x == MIN_IDX_BOARD:
                action_mask[0] = 0  # Block left movement
            elif self.prisoner_x == MAX_IDX_BOARD:
                action_mask[1] = 0  # Block right movement
            if self.prisoner_y == MIN_IDX_BOARD:
                action_mask[2] = 0  # Block down movement
            elif self.prisoner_y == MAX_IDX_BOARD:
                action_mask[3] = 0  # Block up movement
        else:
            if self.guard_x == MIN_IDX_BOARD:
                action_mask[0] = 0
            elif self.guard_x == MAX_IDX_BOARD:
                action_mask[1] = 0
            if self.guard_y == MIN_IDX_BOARD:
                action_mask[2] = 0
            elif self.guard_y == MAX_IDX_BOARD:
                action_mask[3] = 0

            # Action mask to prevent guard from going over escape cell
            if self.guard_x - 1 == self.escape_x:
                action_mask[0] = 0
            elif self.guard_x + 1 == self.escape_x:
                action_mask[1] = 0
            if self.guard_y - 1 == self.escape_y:
                action_mask[2] = 0
            elif self.guard_y + 1 == self.escape_y:
                action_mask[3] = 0

        return action_mask

    def observe(self, agent: int) -> Dict[str, np.ndarray]:
        if agent == self.agent_name_mapping["prisoner"]:
            observation = np.array(
                [self.prisoner_x, self.prisoner_y, self.guard_x, self.guard_y],
                dtype=np.float32)
        else:
            observation = np.array(
                [self.guard_x, self.guard_y, self.prisoner_x, self.prisoner_y],
                dtype=np.float32)
        action_mask = self._get_action_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def get_move(self, action: int) -> tuple:
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}")

        return self.action_map[action]

    def step(self, action: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, bool], Dict[int, bool], Dict[int, dict]]:
        if (
            self.terminations[self.agent_selection] or
            self.truncations[self.agent_selection]
        ):
            # self.agent_selection = self._agent_selector.next()
            return self._was_dead_step(action)

        agent_id = self.agent_selection
        action = int(action)
        x_cmd, y_cmd = self.get_move(action)

        if agent_id == self.agent_name_mapping["prisoner"]:
            self.prisoner_x += x_cmd
            self.prisoner_y += y_cmd
        else:
            self.guard_x += x_cmd
            self.guard_y += y_cmd

        # Check termination conditions
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            self.rewards = {0: -1, 1: 1}
            self.terminations = {a: True for a in self.agents}
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            self.rewards = {0: 1, 1: -1}
            self.terminations = {a: True for a in self.agents}
        # Check truncation conditions
        elif self.timestep > 100:
            self.rewards = {0: 0, 1: 0}
            self.truncations = {a: True for a in self.agents}
        else:
            self.rewards = {0: 0, 1: 0}
            self.terminations = {a: False for a in self.agents}

        # Accumulate rewards
        self._accumulate_rewards()

        # Update timestep and agent selection
        self.timestep += 1
        self.agent_selection = self._agent_selector.next()
        print(self.observe(agent_id))

    def render(self):
        """Renders the environment."""
        grid = np.zeros((7, 7))
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.prisoner_x = 0
        self.prisoner_y = 0
        self.guard_x = MAX_IDX_BOARD
        self.guard_y = MAX_IDX_BOARD
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)
        # self.observations = {agent: self.observe(
        #     agent) for agent in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        # return self.observations, self.infos

    def close(self):
        pass
