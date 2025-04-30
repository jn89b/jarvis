

import yaml
import gymnasium as gym
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from ray.rllib.env import MultiAgentEnv
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.envs.simple_agent import (
    SimpleAgent, Pursuer, Evader, PlaneKinematicModel)
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector
from jarvis.envs.tokens import KinematicIndex
from jarvis.algos.pro_nav import ProNavV2
from jarvis.envs.multi_agent_env import AbstractKinematicEnv
import copy
import itertools
# abstract methods
from abc import ABC, abstractmethod

"""
Environment where
one single agent has:
    pursuer and evader policy
N agents with pursuer policies
I need to change the name of the policies and remap

"""

OFFENSIVE_IDX: int = 0
DEFENSIVE_IDX: int = 1


class HRLMultiAgentEnv(AbstractKinematicEnv):
    """
    https://discuss.ray.io/t/im-confused-about-how-policy-mapping-works-in-configuration/7001/2
    I need to change the name of the policies and remap correctly:
        - Good guy:
            - Evader policies
            - Pursuer policies
        - Bad guys
            - Pursuer policies

    Environment ends when:
        - Good guy is captured by bad guys
        - Good guy captures the target location

    Observation space should be formated as follows:

    Right now for my good guy he's going to have two locations

    agent_obs_spaces = {
        "good_guy_hrl": gym.spaces.Box(low=low_high_obs, high=high_high_obs, dtype=np.float32),
        "good_guy_offensive": gym.spaces.Dict({
            "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(
                                                self.action_spaces["low_attack_agent"].n,),
                                            dtype=np.float32)
        }),
        "good_guy_defensive": gym.spaces.Dict({
            "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(
                                                self.action_spaces["low_avoid_agent"].n,),
                                            dtype=np.float32)
        pursuer_n": gym.spaces.Dict({
            "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(
                                                self.action_spaces["low_pursuer_one"].n,),
                                            dtype=np.float32)

        })
    }

    We need to morph the evader agent into a hrl policy

    """

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        super(HRLMultiAgentEnv, self).__init__(config=config)
        self.interaction_config: Dict[str,
                                      Any] = self.agent_config['interaction']
        self.relative_state_observations: Dict[str,
                                               Any] = self.agent_config['relative_state_observations']
        # TODO: This is super hacky, but I have to do this for now
        # Have to call out the action spaces and observation spaces
        # because its a requirement for the environment class to work
        self.action_spaces: Dict[str,
                                 gym.spaces.Dict] = self.init_action_space()

        self.observation_spaces: Dict[str,
                                      gym.spaces.Dict] = self.init_observation_space()

        self.good_guy_hrl_key: str = "good_guy_hrl"  # high level representation agent
        self.good_guy_offensive_key: str = "good_guy_offensive"
        self.good_guy_defensive_key: str = "good_guy_defensive"

        self.terminal_reward: float = 1000.0
        self.all_done_step: int = 0
        self.use_pronav: bool = True
        # we have to override this agent_cycle to include the good guy
        self.last_high_level_action: int = 0
        assert self.battlespace is not None
        self.__init__agents()
        self.replace_evader_with_good_guy()
        self.insert_target()
        self.current_step: int = 0
        self.possible_agents: List[int] = self.agents
        self.agent_cycle = itertools.cycle(self.possible_agents)
        self.current_agent: str = next(self.agent_cycle)
        self.old_distance_from_target: float = None
        # self.agents: List[int] = [
        #     str(agent.agent_id) for agent in self.get_controlled_agents]

    def __init__agents(self) -> None:
        """
        Randomly spawns agents in the environment

        Right now for our case we are going to center
        this agent in the center of the battlespace with
        some perturbation
        TODO: Refactor and put into a strategy pattern
        """
        self.battlespace.clear_agents()
        num_evaders: int = self.agent_config['num_evaders']
        num_pursuers: int = self.agent_config['num_pursuers']
        current_agent_id = 0
        is_evader_controlled: bool = self.agent_config['evaders']['is_controlled']
        is_pursuer_controlled: bool = self.agent_config['pursuers']['is_controlled']

        current_agent_id: int = self.spawn_agents(
            num_agents=num_evaders,
            agent_id=current_agent_id,
            is_pursuer=False,
            is_controlled=is_evader_controlled)

        current_agent_id: int = self.spawn_pursuers(
            num_pursuers=num_pursuers,
            agent_id=current_agent_id,
            is_controlled=is_pursuer_controlled,
            evader=self.get_evader_agents()[0]
        )

    def spawn_pursuers(self, num_pursuers: int, agent_id: str,
                       is_controlled: bool, evader: SimpleAgent) -> int:
        """
        Args:
            num_pursuers (int): Number of pursuers to spawn
            agent_id (int): The id of the agent to spawn
            is_controlled (bool): True if the agent is controlled, False otherwise
            evader (SimpleAgent): The evader to capture
        Returns:
            agent_id (int): The id of NEXT agent to spawn
        """

        min_radius_spawn: float = self.spawn_config['distance_from_other_agents']['min']
        max_radius_spawn: float = self.spawn_config['distance_from_other_agents']['max']
        state_limits: Dict[str, Any] = self.pursuer_state_limits

        for i in range(num_pursuers):
            random_heading: float = np.random.uniform(-np.pi, np.pi)
            random_radius: float = np.random.uniform(
                min_radius_spawn, max_radius_spawn)
            x_pos: float = evader.state_vector.x + \
                random_radius * np.cos(random_heading)

            y_pos: float = evader.state_vector.y + \
                random_radius * np.sin(random_heading)
            z_pos: float = np.random.uniform(
                self.pursuer_state_limits['z']['min']+20,
                self.pursuer_state_limits['z']['max']-20)

            dx: float = evader.state_vector.x - x_pos
            dy: float = evader.state_vector.y - y_pos

            heading = np.arctan2(dy, dx) + np.random.uniform(-np.pi/4, np.pi/4)
            # wrap heading between -pi and pi
            heading = (heading + np.pi) % (2 * np.pi) - np.pi

            rand_velocity = np.random.uniform(
                state_limits['v']['min'],
                state_limits['v']['max'])

            rand_phi = np.random.uniform(state_limits['phi']['min'],
                                         state_limits['phi']['max'])
            rand_theta = np.random.uniform(state_limits['theta']['min'],
                                           state_limits['theta']['max'])
            state_vector = StateVector(
                x=x_pos, y=y_pos, z=z_pos, roll_rad=rand_phi,
                pitch_rad=rand_theta, yaw_rad=heading, speed=rand_velocity)
            plane_model: PlaneKinematicModel = PlaneKinematicModel()
            radius_bubble: float = self.agent_config['interaction']['bubble_radius']
            capture_radius: float = self.agent_config['interaction']['capture_radius']
            agent: Pursuer = Pursuer(
                battle_space=self.battlespace,
                state_vector=state_vector,
                simple_model=plane_model,
                radius_bubble=radius_bubble,
                is_controlled=is_controlled,
                agent_id=str(agent_id),
                capture_radius=capture_radius)

            self.insert_agent(agent)
            agent_id += 1

        return agent_id

    def replace_evader_with_good_guy(self) -> None:
        """
        TODO:
            The evader agent becomes the good guy
            I'm going to need to replace the list of controlled
            agents with my good guy stuff
            I need to add an additional case for observe method

        Good guy offensive and defensive will recieve the same
        state vector the evader just need to replace the ID

        We also need to slot in good_guy_hrl in the agents list
        """

        # Updating the observation space with the good guy
        high_level_obs_space: Dict[str, gym.spaces.Dict] = {
            self.good_guy_hrl_key: self.good_guy_obs_space(),
        }

        high_level_obs_space[self.good_guy_offensive_key] = self.good_guy_offensive_obs_space(
        )
        high_level_obs_space[self.good_guy_defensive_key] = self.good_guy_defensive_obs_space(
        )

        for k, v in high_level_obs_space.items():
            # insert to observation space
            self.observation_spaces[k] = v

        # Updating the action space for the good guy
        good_guy_action_space = {}
        hrl_action_space: gym.spaces.Dict = gym.spaces.Dict({
            "action": gym.spaces.Discrete(n=2)
        })
        offensive_action_space: gym.spaces.Dict = gym.spaces.Dict({
            "action": self.get_discrete_action_space(is_pursuer=True)
        })
        defensive_action_space: gym.spaces.Dict = gym.spaces.Dict({
            "action": self.get_discrete_action_space(is_pursuer=False)
        })
        good_guy_action_space[self.good_guy_hrl_key] = hrl_action_space
        good_guy_action_space[self.good_guy_offensive_key] = offensive_action_space
        good_guy_action_space[self.good_guy_defensive_key] = defensive_action_space

        for k, v in good_guy_action_space.items():
            self.action_spaces[k] = v

        # these guys need to be coupled together
        evader: Evader = self.get_evader_agents()[0]
        evader.is_controlled = True
        self.agents.remove(evader.agent_id)
        self.remove_agent(evader.agent_id)
        # create a copy of the evader
        good_guy_hrl: Evader = copy.deepcopy(evader)
        good_guy_hrl.agent_id = self.good_guy_hrl_key
        self.insert_agent(good_guy_hrl)
        self.agents = [agent.agent_id for agent in self.get_all_agents]

        self.agents.append(self.good_guy_offensive_key)
        self.agents.append(self.good_guy_defensive_key)

        # remove the observation and action space for the evader
        del self.observation_spaces[evader.agent_id]
        del self.action_spaces[evader.agent_id]

    def update_good_guy(self) -> None:
        """
        """
        # these guys need to be coupled together
        evader: Evader = self.get_evader_agents()[0]
        evader.is_controlled = True
        self.remove_agent(evader.agent_id)
        # create a copy of the evader
        good_guy_hrl: Evader = copy.deepcopy(evader)
        good_guy_hrl.agent_id = self.good_guy_hrl_key
        self.insert_agent(good_guy_hrl)
        self.agents = [agent.agent_id for agent in self.get_all_agents]

        self.agents.append(self.good_guy_offensive_key)
        self.agents.append(self.good_guy_defensive_key)

    def adjust_pitch(self, selected_agent: Pursuer,
                     evader: SimpleAgent,
                     action: Dict[str, Any],
                     target_instead: bool = False,
                     target_statevector: StateVector = None) -> Dict[str, np.ndarray]:
        """

        Args:
            selected_agent (Pursuer): The pursuer agent
            evader (SimpleAgent): The evader agent
        Returns:
            Dict[str, np.ndarray]: The adjusted action for the pursuer agent
        """
        pitch_idx: int = 0
        if target_instead:
            dz = selected_agent.state_vector.z - target_statevector.z
        else:
            dz: float = selected_agent.state_vector.z - evader.state_vector.z
        distance: float = selected_agent.state_vector.distance_2D(
            evader.state_vector)
        pitch_cmd: float = np.arctan2(dz, distance)
        max_pitch: float = self.pursuer_control_limits['u_theta']['max']
        min_pitch: float = self.pursuer_control_limits['u_theta']['min']
        pitch_cmd = np.clip(pitch_cmd, min_pitch, max_pitch)
        action[pitch_idx] = -pitch_cmd

        return action

    def insert_target(self) -> None:
        """
        Spawns the target in the environment
        based on configuration file -> refer to simple_env_config.yaml
        """
        self.target_config: Dict[str, Any] = self.spawn_config['target']
        randomize: bool = self.target_config['randomize']
        spawn_radius: float = self.target_config['spawn_radius_from_agent']

        evader: Evader = self.get_evader_agents()[0]
        if evader is None:
            raise ValueError("Evader agent is not defined")

        if randomize:
            min_radius: float = spawn_radius['min']
            max_radius: float = spawn_radius['max']
            rand_radius: float = np.random.uniform(min_radius, max_radius)
            sign_convention: int = np.random.choice([-1, 1])
            rand_radius *= sign_convention
            x: float = evader.state_vector.x + \
                rand_radius
            y: float = evader.state_vector.y + \
                rand_radius
            z_bounds: List[float] = self.battlespace.z_bounds
            z: float = np.random.uniform(z_bounds[0]+10, z_bounds[1]-10)

        else:
            x: float = self.target_config['position']['x']
            y: float = self.target_config['position']['y']
            z: float = self.target_config['position']['z']

        target_vector: StateVector = StateVector(
            x=x, y=y, z=z, yaw_rad=0, roll_rad=0, pitch_rad=0, speed=0)

        self.target: StateVector = target_vector

    def good_guy_obs_space(self) -> gym.spaces.Box:
        """
        This observation space will have consist of:
        - euclidean distance from target
        - n number euclidean distance from the bad guys
        """
        distance_from_target_min: float = 0.0
        distance_from_target_max: float = 10000.0

        distance_from_pursuer_min: float = 0.0
        distance_from_pursuer_max: float = 10000.0

        n_pursuers: int = self.agent_config['num_pursuers']

        total_index: int = n_pursuers + 1

        low_obs: List[float] = []
        high_obs:  List[float] = []

        low_obs.append(distance_from_target_min)
        high_obs.append(distance_from_target_max)

        for i in range(total_index-1):
            low_obs.append(distance_from_pursuer_min)
            high_obs.append(distance_from_pursuer_max)

        obs_space = gym.spaces.Box(low=np.array(low_obs),
                                   high=np.array(high_obs),
                                   dtype=np.float32)
        action_mask = gym.spaces.Box(low=0.0, high=1.0,
                                     shape=(
                                         2,),
                                     dtype=np.float32)

        return gym.spaces.Dict({
            "observations": obs_space,
            "action_mask": action_mask
        })

    def good_guy_offensive_obs_space(self) -> gym.spaces.Dict:
        """
        Returns an offensive observation space based
        which is the same thing as the pursuer's 
        observation space
        """
        pursuer: Pursuer = self.get_pursuer_agents()[0]
        obs_space: Dict[str, Any] = self.observation_spaces[pursuer.agent_id]

        return obs_space

    def good_guy_defensive_obs_space(self) -> gym.spaces.Box:
        """
        Returns a defensive observation space based on the
        evader observation space
        """
        evader: Evader = self.get_evader_agents()[0]
        # action_sum = self.action_spaces[evader.agent_id]["action"].nvec.sum()

        obs_space: Dict[str, Any] = self.observation_spaces[evader.agent_id]

        return obs_space

    def init_action_space(self) -> Dict[str, gym.spaces.Dict]:
        """
        Returns the action space for the pursuers and evaders
        """

        self.action_spaces: Dict[str, gym.spaces.Dict] = {}
        for agent in self.get_controlled_agents:
            # Get the original MultiDiscrete action space.
            multi_action_space = self.get_discrete_action_space(
                is_pursuer=agent.is_pursuer)
            # Wrap it in a Dict.
            action_space = gym.spaces.Dict({"action": multi_action_space})
            self.action_spaces[agent.agent_id] = action_space

        return self.action_spaces

    def init_observation_space(self) -> Dict[str, gym.spaces.Dict]:
        """
        Returns the observation space for the pursuers and evaders

        Dictionary is the in the following format of the form:
        {
            'agent_id': {
                'observations': gym.spaces.Box,
                'action_mask': gym.spaces.MultiBinary
            }
            'agent_id': {
                'observations': gym.spaces.Box,
                'action_mask': gym.spaces.MultiBinary
            }
        }
        """
        self.observation_spaces: Dict[str, gym.spaces.Dict] = {}
        for agent in self.get_controlled_agents:
            num_actions = self.action_spaces[agent.agent_id]["action"].nvec.sum(
            )

            if num_actions is None:
                raise ValueError(
                    "The number of actions for the agent is not defined")

            observation_space: gym.spaces.Dict = self.build_observation_space(
                is_pursuer=agent.is_pursuer,
                num_actions=num_actions)

            # we are going to need to include the relative position of the
            # agent to the other agents
            obs = observation_space['observations']
            relative_positions: Dict[str,
                                     Any] = self.relative_state_observations['position']
            relative_velocities: Dict[str,
                                      Any] = self.relative_state_observations['velocity']
            relative_heading: Dict[str,
                                   Any] = self.relative_state_observations['heading']

            x_low: float = relative_positions['x']['low']
            y_low: float = relative_positions['y']['low']
            z_low: float = relative_positions['z']['low']

            x_high: float = relative_positions['x']['high']
            y_high: float = relative_positions['y']['high']
            z_high: float = relative_positions['z']['high']

            vel_low: float = relative_velocities['low']
            vel_high: float = relative_velocities['high']

            heading_low: float = relative_heading['low']
            heading_high: float = relative_heading['high']

            obs_low: np.array = obs.low
            obs_high: np.array = obs.high

            obs_relative_low: List[float] = []
            obs_relative_high: List[float] = []
            # check how many other agents are in the environment
            for other_agent in self.get_controlled_agents:
                if agent.is_pursuer == other_agent.is_pursuer:
                    continue

                if other_agent.agent_id == agent.agent_id:
                    continue

                obs_relative_low.extend([x_low, y_low, z_low,
                                         vel_low, heading_low])
                obs_relative_high.extend([x_high, y_high, z_high,
                                          vel_high, heading_high])

            obs_low = np.concatenate([obs_low, obs_relative_low])
            obs_high = np.concatenate([obs_high, obs_relative_high])
            obs = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            observation_space['observations'] = obs

            self.observation_spaces[agent.agent_id] = observation_space

        return self.observation_spaces

    def observe(self, agent, total_actions=None,
                use_low_level: bool = False) -> Dict[str, Any]:
        """
        TODO: UPDATE
        Need to check if agent has hrl
        if so:
            Observation becomes the euclidean distance from the target
            as well as euclidean distances from the bad guys
        if contains good_guy_offensive:
            - then we play it as pursuer
        if contains good_guy_defensive:
            - then we play it as evader

        Good guy HRL:
            - euclidean distance from target
            - euclidean distance from the bad guys


        Returns the observation for the agent in the environment
        formatted as follows:
        Dict[str, np.array] = {
            'observations': gym.spaces.Box,
            'action_mask': gym.spaces.MultiBinary
        }

        """
        # return super().observe(agent, total_actions)

        # TODO: Probably a better way to do this but this is a quick fix
        if agent is None:
            raise ValueError("Agent is not defined")

        if agent.agent_id == self.good_guy_hrl_key and not use_low_level:
            return self.get_hrl_observation()

        if agent.agent_id == self.good_guy_hrl_key and use_low_level:
            # get the index val
            observation: Dict[str, np.array] = super().observe(
                agent, total_actions)
            obs: np.array = observation['observations']
            masks: np.array = observation['action_mask']

            if self.last_high_level_action == OFFENSIVE_IDX:
                # return the euclidean distance from the the target
                # return the pursuer observation
                target: StateVector = self.target
                relative_pos: np.array = agent.state_vector.array - \
                    target.array
                relative_vel: np.array = agent.state_vector.speed - \
                    target.speed
                relative_heading: np.array = agent.state_vector.yaw_rad - \
                    target.yaw_rad
                relative_pos = relative_pos[:3]
                relative_info: List[float] = [
                    relative_pos[0], relative_pos[1], relative_pos[2],
                    relative_vel, relative_heading]
                obs = np.concatenate([obs, relative_info]).astype(np.float32)

                if obs.shape[0] != self.observation_spaces[self.good_guy_offensive_key]['observations'].shape[0]:
                    raise ValueError("The observation space is not the same \
                        current shape and actual shape is", obs.shape, self.observation_spaces[self.good_guy_offensive_key]['observations'].shape)

            elif self.last_high_level_action == DEFENSIVE_IDX:

                overall_relative_pos: List[float] = []
                for other_agent in self.get_controlled_agents:
                    if agent.agent_id == other_agent.agent_id:
                        continue

                    if agent.is_pursuer == other_agent.is_pursuer:
                        continue

                    relative_pos: np.ndarray = agent.state_vector.array - \
                        other_agent.state_vector.array
                    relative_velocity = agent.state_vector.speed - \
                        other_agent.state_vector.speed
                    relative_heading = agent.state_vector.yaw_rad - \
                        other_agent.state_vector.yaw_rad
                    # wrap heading between -pi and pi
                    relative_heading = (
                        relative_heading + np.pi) % (2 * np.pi) - np.pi

                    relative_pos = relative_pos[:3]

                    relative_info = [relative_pos[0], relative_pos[1], relative_pos[2],
                                     relative_velocity, relative_heading]

                    overall_relative_pos.extend(relative_info)
                    
                obs = np.concatenate(
                    [obs, overall_relative_pos]).astype(np.float32)

                # check if pitch is outside the bounds

                if obs.shape[0] != self.observation_spaces[self.good_guy_defensive_key]['observations'].shape[0]:
                    raise ValueError("The observation space is not the same \
                        current shape and actual shape is", obs.shape, self.observation_spaces[self.good_guy_defensive_key]['observations'].shape)
                
                if masks.shape[0] != self.observation_spaces[self.good_guy_defensive_key]['action_mask'].shape[0]:
                    raise ValueError("The action mask is not the same \
                        current shape and actual shape is", masks.shape, self.action_spaces[agent.agent_id]['action'].nvec.sum())
                    
            observation['observations'] = obs
            return observation

        else:
            # return the pursuer observation
            observation: Dict[str, np.array] = super().observe(
                agent, total_actions)
            obs: np.array = observation['observations']
            masks: np.array = observation['action_mask']

            evader: SimpleAgent = self.get_evader_agents()[0]
            relative_pos: np.array = agent.state_vector.array - \
                evader.state_vector.array
            relative_vel: np.array = agent.state_vector.speed - \
                evader.state_vector.speed
            relative_heading: np.array = agent.state_vector.yaw_rad - \
                evader.state_vector.yaw_rad
            # wrap heading between -pi and pi
            relative_heading = (
                relative_heading + np.pi) % (2 * np.pi) - np.pi

            relative_pos = relative_pos[:3]

            relative_info: List[float] = [
                relative_pos[0], relative_pos[1], relative_pos[2],
                relative_vel, relative_heading]

            obs = np.concatenate([obs, relative_info]).astype(np.float32)

            if obs.shape[0] != self.observation_spaces[agent.agent_id]['observations'].shape[0]:
                raise ValueError("The observation space is not the same \
                    current shape and actual shape is", obs.shape, self.observation_spaces[agent.agent_id]['observations'].shape)

            # check action mask is correct
            if masks.shape[0] != self.observation_spaces[self.good_guy_defensive_key]['action_mask'].shape[0]:
                raise ValueError("The action mask is not the same \
                    current shape and actual shape is", masks.shape, self.action_spaces[agent.agent_id]['action'].nvec.sum())

            observation['observations'] = obs
            
            return observation

    def get_hrl_observation(self) -> Dict[str, np.array]:
        """
        Args:
            agent: SimpleAgent
        Returns:
            observation: np.array
            Formatted as follows:
                - euclidean distance from the target
                - euclidean distances for each of the pursuers
        """
        obs: List[float] = []
        good_guy: SimpleAgent = self.get_specific_agent(self.good_guy_hrl_key)
        target: StateVector = self.target
        # euclidean distance from the target
        target_distance: float = good_guy.state_vector.distance_3D(target)

        obs.append(target_distance)
        for pursuer in self.get_pursuer_agents():
            distance: float = good_guy.state_vector.distance_3D(
                pursuer.state_vector)
            obs.append(distance)

        obs = np.array(obs, dtype=np.float32)
        # num actions is 2 for the HRL
        num_actions: int = 2
        valid_actions = np.ones(num_actions,
                                dtype=np.float32)
        observations: Dict[str, np.array] = {
            "observations": obs,
            "action_mask": valid_actions}

        actual_obs_shape = self.observation_spaces[self.good_guy_hrl_key]['observations'].shape
        if observations['observations'].shape[0] != actual_obs_shape[0]:
            raise ValueError("The observation space is not the same \
                current shape and actual shape is", observations['observations'].shape, actual_obs_shape)

        return observations

    def step_hrl_policy(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        If we stepping thruogh the HRL policy then we need to make sure
        the next agent returned is the good guy offensive or defensive

        Process is as follows:
        - Set the action
        - Simulate the action
        - Return the observations, rewards, terminateds, truncateds, infos

        The next agent should always be one of the policies
        """
        # observations: Dict[str, np.array] = self.get_hrl_observation()
        action_val = action[self.good_guy_hrl_key]['action']
        self.last_high_level_action = action_val
        agent: Evader = self.get_specific_agent(self.good_guy_hrl_key)
        # TODO: Super hacky but need to get the number of actions
        pursuer: Pursuer = self.get_pursuer_agents()[0]
        total_actions = self.action_spaces[pursuer.agent_id]["action"].nvec.sum(
        )
        observations: Dict[str, np.array] = {}

        if action_val == OFFENSIVE_IDX:
            # return the offensive observation
            observations[self.good_guy_offensive_key] = self.observe(
                agent=agent, total_actions=total_actions,
                use_low_level=True)
            self.current_agent = self.good_guy_offensive_key
        elif action_val == DEFENSIVE_IDX:
            observations[self.good_guy_defensive_key] = self.observe(
                agent=agent, total_actions=total_actions,
                use_low_level=True)
            self.current_agent = self.good_guy_defensive_key
        else:
            raise ValueError("High level action is not defined \
                value is {}".format(action_val))

        if not self.valid_observations(observations, self.current_agent):
            raise ValueError("Observations are not defined")

        return observations

    def step_low_level_policy(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        If we are stepping through the low level policy than
        we need to return the next pursuer and step thruogh

        Process is as follows:
        - Set the action
        - Simulate the action
        - Return the observations, rewards, terminateds, truncateds, infos

        The next agent should always be the pursuer agent
        """
        observations: Dict[str, np.array] = {}

        if self.last_high_level_action == OFFENSIVE_IDX:
            ## Set the action and simulate ###
            selected_agent: SimpleAgent = self.get_specific_agent(
                self.good_guy_hrl_key)
            # super hacky but need to get the number of actions from the pursuer
            # since they are the same
            pursuer_id: str = self.get_pursuer_agents()[0].agent_id
            num_actions: int = self.action_spaces[pursuer_id]["action"].nvec.sum(
            )
            
            # processing the actions
            if self.use_pronav:
                pronav: ProNavV2 = ProNavV2()
                current_pos = selected_agent.state_vector.array[0:3]
                target_pos = self.target.array[0:3]
                relative_pos = target_pos - current_pos
                relative_vel = self.target.speed - \
                    selected_agent.state_vector.speed
                action_cmd: np.array = self.discrete_to_continuous_action(
                    action[str(self.good_guy_offensive_key)]['action'])
                # get the velocity of the agent
                vel_cmd = action_cmd[-1]
                if vel_cmd >= 30:
                    vel_cmd = 30
                action_cmd = pronav.predict(
                    current_pos=current_pos,
                    relative_pos=relative_pos,
                    current_heading=selected_agent.state_vector.yaw_rad,
                    current_speed=selected_agent.state_vector.speed,
                    relative_vel=relative_vel,
                    dont_predict=True,
                    consider_yaw=False,
                    max_vel=vel_cmd
                )
                #clip the dz command
                # delta_z = target_pos - current_pos
                action_cmd[0] = np.clip(action_cmd[0],
                                        self.pursuer_control_limits['u_dz']['min'],
                                        self.pursuer_control_limits['u_dz']['max'])
            # else:
            #     action_cmd: np.array = self.discrete_to_continuous_action(
            #         action[str(self.good_guy_offensive_key)]['action'])
            #     if selected_agent.is_pursuer:
            #         action_cmd = self.adjust_pitch(
            #             selected_agent, self.get_evader_agents()[
            #                 0], action_cmd,
            #             target_instead=True, target_statevector=self.target)

            command_action: Dict[str, np.array] = {
                selected_agent.agent_id: action_cmd}
            self.simulate(command_action, use_multi=True)

            # Anytime we make a low level action we need to switch to the pursuer 
            # to make sure time is being stepped correctly
            # we have to use the agent cycle to get the next agent otherwise 
            # we will get stuck in a loop
            while self.current_agent != self.get_pursuer_agents()[0].agent_id:
                self.current_agent = next(self.agent_cycle)
                
            agent = self.get_specific_agent(self.current_agent)
            if agent is None:
                raise ValueError("The agent is not defined")
            observations[self.current_agent] = self.observe(
                agent=agent, total_actions=num_actions)

        elif self.last_high_level_action == DEFENSIVE_IDX:
            # we want the observation to return the good guy states and
            # the relative positions of the threats from us
            ## Set the action and simulate ###
            selected_agent: SimpleAgent = self.get_specific_agent(
                self.good_guy_hrl_key)
            # super hacky but need to get the number of actions from the pursuer
            # since they are the same
            pursuer_id: str = self.get_pursuer_agents()[0].agent_id
            num_actions: int = self.action_spaces[pursuer_id]["action"].nvec.sum(
            )
            action_cmd: np.array = self.discrete_to_continuous_action(
                action[self.good_guy_defensive_key]['action'])

            command_action: Dict[str, np.array] = {
                selected_agent.agent_id: action_cmd}
            self.simulate(command_action, use_multi=True)

            # I'm going to need to return the observation of the next pursuer
            # Reward everything else here
            self.current_agent: str = self.get_pursuer_agents()[0].agent_id
            num_actions = self.action_spaces[self.current_agent]["action"].nvec.sum(
            )
            agent: Pursuer = self.get_specific_agent(self.current_agent)
            observations = {}
            observations[agent.agent_id] = self.observe(
                agent=agent, total_actions=num_actions)
        else:
            raise ValueError("High level action is not defined")

        if not self.valid_observations(observations, self.current_agent):
            raise ValueError("Observations are not defined")

        return observations

    def valid_observations(self, observations: Dict[str, np.array],
                           agent_id: str) -> None:
        """
        Check if the observations are defined
        """
        if not observations:
            print("Observations are not defined")
            return False

        if agent_id not in observations:
            print("Agent ID is not in the observations",
                  observations)
            return False

        return True

    def step_pursuer_policy(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        If we are stepping through the low level policy than
        we need to return the next pursuer and step thruogh

        Process is as follows:
        - Set the action
        - Simulate the action
        - Return the observations, rewards, terminateds, truncateds, infos

        The next agent should be the next agent called in the cycle
        """
        observations: Dict[str, np.array] = {}

        selected_agent: SimpleAgent = self.get_specific_agent(
            self.current_agent)
        # super hacky but need to get the number of actions from the pursuer
        # since they are the same
        pursuer_id: str = self.get_pursuer_agents()[0].agent_id
        num_actions: int = self.action_spaces[pursuer_id]["action"].nvec.sum(
        )
        action_cmd: np.array = self.discrete_to_continuous_action(
            action[str(selected_agent.agent_id)]['action'])
        # action_cmd = self.adjust_pitch(
        #     selected_agent, self.get_evader_agents()[0], action_cmd)

        if self.use_pronav:
            pronav: ProNavV2 = ProNavV2()
            current_pos = selected_agent.state_vector.array[0:3]
            evader: Evader = self.get_evader_agents()[0]
            target_pos = evader.state_vector.array[0:3]
            relative_pos = target_pos - current_pos
            relative_vel = evader.state_vector.speed - \
                selected_agent.state_vector.speed
            action_cmd = pronav.predict(
                current_pos=current_pos,
                relative_pos=relative_pos,
                current_heading=selected_agent.state_vector.yaw_rad,
                current_speed=selected_agent.state_vector.speed,
                relative_vel=relative_vel
            )
            #clip the dz command
            action_cmd[0] = np.clip(action_cmd[0], 
                                    self.pursuer_control_limits['u_dz']['min'],
                                    self.pursuer_control_limits['u_dz']['max'])


        command_action: Dict[str, np.array] = {
            selected_agent.agent_id: action_cmd}

        self.simulate(command_action, use_multi=True)

        # I'm going to need to return the observation of the next pursuer
        # TODO: make sure I am cycling this correctly
        observations = {}
        self.current_agent: str = next(self.agent_cycle)

        # Using this in case we have the same cycle
        # while self.current_agent == selected_agent.agent_id:
        #     self.current_agent: str = next(self.agent_cycle)
        #     print("Current agent is", self.current_agent)
        if (self.current_agent == self.good_guy_hrl_key or
            self.current_agent == self.good_guy_offensive_key or
                self.current_agent == self.good_guy_defensive_key):
            self.current_agent = self.good_guy_hrl_key
            observations[self.current_agent] = self.get_hrl_observation()
        else:
            num_actions = self.action_spaces[self.current_agent]["action"].nvec.sum(
            )
            if num_actions is None:
                raise ValueError(
                    "The number of actions for the agent is not defined")

            agent: SimpleAgent = self.get_specific_agent(self.current_agent)
            if agent is None:
                raise ValueError("The agent is not defined")
            observations: Dict[str, np.array] = {}
            observations[agent.agent_id] = self.observe(
                agent=agent, total_actions=num_actions)

        if not self.valid_observations(observations, self.current_agent):
            raise ValueError("Observations are not defined")

        actual_obs_shape = self.observation_spaces[self.current_agent]['observations'].shape
        if observations[self.current_agent]['observations'].shape[0] != actual_obs_shape[0]:
            raise ValueError("The observation space is not the same \
                current shape and actual shape is", observations[self.current_agent]['observations'].shape, actual_obs_shape)

        # compute the rewards here
        return observations

    def step(self, action_dict: Dict[str, Any],
             specific_agent_id: int = None) -> \
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Remember for multi-agent environments
        we need to conduct the action of the current agent
        then move to the next agent
        then return the observations, rewards, terminateds, truncateds, infos of that next agent
        """
        # terminateds: Dict[str, bool] = {"__all__": False}
        # truncateds: Dict[str, bool] = {"__all__": False}
        # rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        # infos: Dict[str, Any] = {}
        # observations: Dict[str, np.array] = {}
        use_low_level_policy: bool = False
        # Since we don't have an actual agent for the offensive and
        # defensive policies hook it to the good guy agent
        # Selecing the agent to act
        if specific_agent_id is None:
            if (self.current_agent == self.good_guy_offensive_key or
                    self.current_agent == self.good_guy_defensive_key):
                selected_agent: SimpleAgent = self.get_specific_agent(
                    self.good_guy_hrl_key)
                use_low_level_policy = True
            else:
                selected_agent: SimpleAgent = self.get_specific_agent(
                    self.current_agent)
        else:
            if (specific_agent_id == self.good_guy_offensive_key or
                    specific_agent_id == self.good_guy_defensive_key):
                selected_agent: SimpleAgent = self.get_specific_agent(
                    self.good_guy_hrl_key)
                use_low_level_policy = True
            else:
                selected_agent: SimpleAgent = self.get_specific_agent(
                    specific_agent_id)

        ####  PROCESS THE ACTION AND STEP THROUGH ####
        # check if we are using low level policy
        # if so then we need to hook it to last_action
        if use_low_level_policy:
            next_observations: Dict[str, Any] = self.step_low_level_policy(
                action_dict)
        elif selected_agent.agent_id == self.good_guy_hrl_key:
            next_observations: Dict[str,
                                    Any] = self.step_hrl_policy(action_dict)
        else:
            next_observations: Dict[str,
                                    Any] = self.step_pursuer_policy(action_dict)

        rewards, terminateds, truncateds = self.rewards_truncated_terminated()
        infos: Dict[str, Any] = {}

        self.all_done_step += 1
        # this is a simple step counter to make sure all agents have taken a step
        if self.all_done_step >= len(self.agents):
            self.all_done_step = 0
            self.current_step += 1

        if not next_observations:
            raise ValueError("Observations are not defined")

        return next_observations, rewards, terminateds, truncateds, infos

    def sigmoid(self, x: float) -> float:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def reward_heading_and_delta(self, dot_product: float,
                                 delta_distance: float,
                                 alpha: float = 5.0,
                                 beta: float = 12) -> float:
        """
        Args:
            dot_product (float): The dot product between the current heading and desired heading
            delta_distance (float): The change in distance between the pursuer and the evader

        Returns:
            float: The reward from the dot product and the change in distance

        The reward is computed as the sigmoid of the dot product
        and the sigmoid of the change in distance

        Dot products close to 1 will be rewarded higher
        Positive change in distance will be rewarded higher
        """
        sigmoid_dot: float = self.sigmoid(alpha*dot_product)
        sigmoid_distance: float = self.sigmoid(beta*delta_distance)

        reward: float = sigmoid_dot * sigmoid_distance
        return reward

    def compute_pursuer_reward(self, pursuer: Pursuer, evader: Evader,
                               update: bool = True) -> float:
        """
        Compute the distance between the pursuer and the evader
        Reward for closing the distance between the pursuer and the evader

        Get the dot product between the current heading and desired heading
        """
        pursuer_unit_vec: StateVector = pursuer.state_vector.unit_vector_2D()
        pursuer_unit_vec = np.array([pursuer_unit_vec.x, pursuer_unit_vec.y])

        los = np.array([evader.state_vector.x - pursuer.state_vector.x,
                        evader.state_vector.y - pursuer.state_vector.y])
        los_unit: float = los / np.linalg.norm(los)
        dot_product: float = np.dot(pursuer_unit_vec, los_unit)
        # heading_error: float = (heading_error + np.pi) % (2 * np.pi) - np.pi
        distance: float = pursuer.state_vector.distance_3D(evader.state_vector)
        if pursuer.old_distance_from_evader is None:
            delta_distance: float = 100
            if update:
                pursuer.old_distance_from_evader = distance
            return 0
        else:
            # old distance from evader - current distance from evader
            # we want to reward for converging to the evader
            # 5 - 3 = 2
            # 3 - 5 = -2
            delta_distance: float = pursuer.old_distance_from_evader - distance

        reward = self.reward_heading_and_delta(
            dot_product=dot_product, delta_distance=delta_distance)

        if update:
            pursuer.old_distance_from_evader = distance

        return reward

    def compute_evader_reward(self, pursuer: Pursuer, evader: Evader) -> float:
        """
        Compute the evader's reward as the negative of the pursuer's reward,
        but without updating the pursuer's state (e.g., old_distance_from_evader).
        """
        # Compute the unit vector for the pursuer's heading.
        pursuer_unit_vec = pursuer.state_vector.unit_vector_2D()
        pursuer_unit_vec = np.array([pursuer_unit_vec.x, pursuer_unit_vec.y])

        # Calculate line-of-sight vector from pursuer to evader.
        los = np.array([
            evader.state_vector.x - pursuer.state_vector.x,
            evader.state_vector.y - pursuer.state_vector.y
        ])
        los_unit = los / np.linalg.norm(los)

        # Dot product between pursuer heading and the line-of-sight unit vector.
        dot_product = np.dot(pursuer_unit_vec, los_unit)

        # Compute the current distance between pursuer and evader.
        distance = pursuer.state_vector.distance_3D(evader.state_vector)

        # Determine the delta distance without updating pursuer.old_distance_from_evader.
        if pursuer.old_distance_from_evader is None:
            # If no previous distance exists, we cannot compute a delta.
            # Return a neutral reward.
            return 0
        else:
            delta_distance = evader.old_distance_from_pursuer - distance

        # Compute the reward using the shared logic.
        reward = self.reward_heading_and_delta(
            dot_product=dot_product,
            delta_distance=delta_distance
        )

        evader.old_distance_from_pursuer = distance

        # Return the negative reward for the evader without causing any state updates.
        return -dot_product #- delta_distance

    def compute_target_reward(self, good_guy: Pursuer, target: StateVector) -> float:
        """
        Compute the reward for the target based on the distance from the good_guy.
        """
        # distance = good_guy.state_vector.distance_3D(target)
        # return -distance
        pursuer_unit_vec: StateVector = good_guy.state_vector.unit_vector_2D()
        pursuer_unit_vec = np.array([pursuer_unit_vec.x, pursuer_unit_vec.y])

        los = np.array([target.x - good_guy.state_vector.x,
                        target.y - good_guy.state_vector.y])
        los_unit: float = los / np.linalg.norm(los)
        dot_product: float = np.dot(pursuer_unit_vec, los_unit)

        if self.old_distance_from_target is None:
            delta_distance: float = 100
            self.old_distance_from_target = good_guy.state_vector.distance_3D(
                target)
            return 0

        else:
            distance: float = good_guy.state_vector.distance_3D(target)
            delta_distance: float = self.old_distance_from_target - distance

        reward = self.reward_heading_and_delta(
            dot_product=dot_product, delta_distance=delta_distance)

        self.old_distance_from_target = distance
        return reward

    def rewards_truncated_terminated(self) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, bool]]:
        """
        Returns the rewards, terminateds, and truncateds for the agents

        Terminal Rewards are as follows:
            If good guy captures the target:
                - Good guy gets a reward of terminal_reward to all good guy policies
                - Bad guys get a reward of -terminal_reward to all pursuers
            If good guy gets captured:
                - Good guy gets a reward of -terminal_reward to all good guy policies
                - Bad guys get a reward of terminal_reward  to all pursuers

            If out of time:
                - Good guy gets a reward of -terminal_reward to all good guy policies
                - Bad guys get a reward of terminal_reward to all pursuers

            If intermediate rewards:
                -  Good guy HRL policy:
                    - Negative distance from the target and positive distance from the pursuers
                -  Good guy Offensive policy:
                    - Negative distance from the target
                -  Good guy Defensive policy:
                    - Negative distance from the pursuers
                -  Pursuers:
                    - Negative distance from the evader
        """

        terminateds: Dict[str, bool] = {"__all__": False}
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        truncateds: Dict[str, bool] = {agent: False for agent in self.agents}

        if self.current_step >= self.max_steps:
            # out of time
            print("Out of time for the environment Good Guy loses")
            rewards[self.good_guy_hrl_key] = -self.terminal_reward
            rewards[self.good_guy_offensive_key] = -self.terminal_reward
            rewards[self.good_guy_defensive_key] = -self.terminal_reward
            for pursuer in self.get_pursuer_agents():
                rewards[pursuer.agent_id] = self.terminal_reward
            terminateds["__all__"] = True
            return rewards, terminateds, truncateds

        # check capture
        good_guy: SimpleAgent = self.get_specific_agent(self.good_guy_hrl_key)
        target: StateVector = self.target
        distance: float = good_guy.state_vector.distance_3D(target)
        # TODO: reconfig this
        capture_distance: float = 20.0
        if distance <= capture_distance:
            print("Good Guy captures the target", distance)
            rewards[self.good_guy_hrl_key] = self.terminal_reward
            rewards[self.good_guy_offensive_key] = self.terminal_reward
            rewards[self.good_guy_defensive_key] = self.terminal_reward
            for pursuer in self.get_pursuer_agents():
                rewards[pursuer.agent_id] = -self.terminal_reward
            terminateds["__all__"] = True
            return rewards, terminateds, truncateds

        # check if the good guy has been captured
        for pursuer in self.get_pursuer_agents():
            pursuer: Pursuer
            distance: float = good_guy.state_vector.distance_3D(
                pursuer.state_vector)
            if distance <= pursuer.capture_radius or good_guy.crashed:
                distance_from_target: float = good_guy.state_vector.distance_3D(
                    target)
                print("Good Guy has been captured distance from target", distance, distance_from_target)
                rewards[self.good_guy_hrl_key] = -self.terminal_reward
                rewards[self.good_guy_offensive_key] = -self.terminal_reward
                rewards[self.good_guy_defensive_key] = -self.terminal_reward
                for pursuer in self.get_pursuer_agents():
                    rewards[pursuer.agent_id] = self.terminal_reward
                terminateds["__all__"] = True
                return rewards, terminateds, truncateds

        # Compute the intermediate rewards
        min_distance: float = 10000
        closet_pursuer: Pursuer = None
        for pursuer in self.get_pursuer_agents():
            pursuer: Pursuer
            distance: float = good_guy.state_vector.distance_3D(
                pursuer.state_vector)
            if distance < min_distance:
                min_distance = distance
                closet_pursuer = pursuer
            reward: float = self.compute_pursuer_reward(pursuer, good_guy)
            rewards[pursuer.agent_id] = reward

        # intermediate rewards
        intermediate_dist_reward: float = self.compute_target_reward(
            good_guy, target)
        intermediate_evader_reward: float = self.compute_evader_reward(
            closet_pursuer, good_guy)
        lambda_1: float = 1.0
        lambda_2: float = 0.1
        rewards[self.good_guy_hrl_key] = (lambda_1*intermediate_dist_reward) + \
            (lambda_2*intermediate_evader_reward) - 0.1

        rewards[self.good_guy_offensive_key] = intermediate_dist_reward
        rewards[self.good_guy_defensive_key] = intermediate_evader_reward

        return rewards, terminateds, truncateds

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        assert self.battlespace is not None
        self.__init__agents()
        self.update_good_guy()
        self.insert_target()
        self.possible_agents: List[int] = self.agents
        self.agent_cycle = itertools.cycle(self.possible_agents)
        self.current_agent = '1'

        agent: SimpleAgent = self.get_specific_agent(self.current_agent)
        num_actions: int = self.action_spaces[agent.agent_id]["action"].nvec.sum(
        )
        observations: Dict[str, np.array] = {}
        observations[agent.agent_id] = self.observe(
            agent=agent, total_actions=num_actions)
        # observations[self.current_agent] = self.observe(
        #     agent=agent, num_actions=num_actions)
        infos = {}

        return observations, infos
