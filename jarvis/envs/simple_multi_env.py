import yaml
import gymnasium as gym
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

from jarvis.envs.simple_agent import (
    SimpleAgent, Pursuer, Evader, PlaneKinematicModel)
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector

# abstract methods
from abc import ABC, abstractmethod


class AbstracKinematicEnv(gym.Env, ABC):
    """

    """

    def __init__(self,
                 config: Optional[Dict]) -> None:
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = config.get("agents")
        self.spawn_config: Dict[str, Any] = config.get("spawn")
        self.simulation_config: Dict[str, Any] = config.get("simulation")

        # these methods will be implemented by the user
        self.roll_commands: np.array = None
        self.pitch_commands: np.array = None
        self.yaw_commands: np.array = None
        self.airspeed_commands: np.array = None

        self.pursuer_control_limits, self.pursuer_state_limits = self.load_limit_config(
            is_pursuer=True)
        self.evader_control_limits, self.evader_state_limits = self.load_limit_config(
            is_pursuer=False)

        self.__init_battlespace()
        assert self.battlespace is not None
        self.__init__agents()
        self.agents: List[int] = [
            agent.agent_id for agent in self.get_controlled_agents]

    @property
    def get_all_agents(self) -> List[SimpleAgent]:
        """
        Returns all the agents in the environment
        """
        return self.battlespace.all_agents

    @property
    def get_controlled_agents(self) -> List[SimpleAgent]:
        """
        Returns all the controlled agents in the environment
        """
        controlled_agents = []
        for agent in self.battlespace.all_agents:
            if agent.is_controlled:
                controlled_agents.append(agent)

        return controlled_agents

    @abstractmethod
    def step(self, action: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """
        An abstract method that defines the step function
        users must implement this method when they inherit from this class
        """
        raise NotImplementedError

    @abstractmethod
    def init_observation_space(self) -> gym.spaces.Dict:
        """
        An abstract method that defines the observation space
        users must implement this method when they inherit from this class
        """
        raise NotImplementedError

    @abstractmethod
    def init_action_space(self) -> gym.spaces.Dict:
        """
        An abstract method that defines the action space
        users must implement this method when they inherit from this class
        """
        raise NotImplementedError

    def __init_battlespace(self) -> None:
        """
        Creates the battlespace for the environment
        Based on the configuration file the user specifies
        this only happens if there is no battlespace already
        defined by the user
        """
        x_bounds: List[float] = self.config['bounds']['x']
        y_bounds: List[float] = self.config['bounds']['y']
        z_bounds: List[float] = self.config['bounds']['z']

        self.battlespace: BattleSpace = BattleSpace(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
        )

    def __init__agents(self) -> None:
        """
        Randomly spawns agents in the environment

        Right now for our case we are going to center
        this agent in the center of the battlespace with
        some perturbation
        TODO:
        """

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

        current_agent_id: int = self.spawn_agents(
            num_agents=num_pursuers,
            agent_id=current_agent_id,
            is_pursuer=True,
            is_controlled=is_pursuer_controlled)

    def spawn_agents(self,
                     num_agents: int,
                     agent_id: float,
                     is_pursuer: bool = False,
                     is_controlled: bool = False) -> int:
        """
        Args:
            num_agents (int): Number of evaders to spawn
            agent_id (int): The id of the agent to spawn
            is_controlled (bool): True if the agent is controlled, False otherwise
        Returns:
            agent_id (int): The id of NEXT agent to spawn

        Spawns Agents in the environment
        """
        if is_pursuer:
            state_limits = self.pursuer_state_limits
        else:
            state_limits = self.evader_state_limits

        for i in range(num_agents):
            state_limits = self.evader_state_limits
            # control_limits = self.evader_control_limits
            rand_x = np.random.uniform(-10, 10)
            rand_y = np.random.uniform(-10, 10)
            rand_z = np.random.uniform(state_limits['z']['min'],
                                       state_limits['z']['max'])

            rand_phi = np.random.uniform(state_limits['phi']['min'],
                                         state_limits['phi']['max'])
            rand_theta = np.random.uniform(state_limits['theta']['min'],
                                           state_limits['theta']['max'])
            rand_psi = np.random.uniform(0, 2 * np.pi)
            rand_velocity = np.random.uniform(
                state_limits['v']['min'],
                state_limits['v']['max'])
            state_vector = StateVector(
                x=rand_x, y=rand_y, z=rand_z, roll_rad=rand_phi,
                pitch_rad=rand_theta, yaw_rad=rand_psi, speed=rand_velocity)

            plane_model: PlaneKinematicModel = PlaneKinematicModel()
            radius_bubble: float = self.agent_config['interaction']['bubble_radius']
            if is_pursuer:
                capture_radius: float = self.agent_config['interaction']['capture_radius']
                agent = Pursuer(
                    battle_space=self.battlespace,
                    state_vector=state_vector,
                    simple_model=plane_model,
                    radius_bubble=radius_bubble,
                    is_controlled=is_controlled,
                    agent_id=agent_id,
                    capture_radius=capture_radius)
            else:
                agent: Evader = Evader(
                    battle_space=self.battlespace,
                    state_vector=state_vector,
                    simple_model=plane_model,
                    radius_bubble=radius_bubble,
                    is_controlled=is_controlled,
                    agent_id=agent_id)

            self.insert_agent(agent)
            agent_id += 1

        return agent_id

    def load_limit_config(self, is_pursuer: bool) -> Tuple[Dict[str, Dict[str, float]],
                                                           Dict[str, Dict[str, float]]]:
        """
        Args:
            is_pursuer (bool): True if the agent is a pursuer, False if the agent is an evader
        Returns:
            A tuple containing the control limits and state limits of the agent

        Loads the configuration of the state limits and control limits
        of a pursuer or evader agent
        Refer to the simple_env_config.yaml file for more information
        """
        control_limits_dict = {}
        state_limits_dict = {}

        if is_pursuer:
            config: Dict[str, Any] = self.agent_config['pursuers']
        else:
            config: Dict[str, Any] = self.agent_config['evaders']

        # Extract control limits
        for key, limits in config.get('control_limits', {}).items():
            control_limits_dict[key] = {
                'min': float(limits['min']),
                'max': float(limits['max'])
            }

        # Extract state limits
        for key, limits in config.get('state_limits', {}).items():
            state_limits_dict[key] = {
                'min': float('-inf') if limits['min'] == '-inf' else float(limits['min']),
                'max': float('inf') if limits['max'] == 'inf' else float(limits['max'])
            }

        return control_limits_dict, state_limits_dict

    def insert_agent(self, agent: SimpleAgent) -> None:
        if self.battlespace.all_agents is None:
            self.battlespace.all_agents: List[SimpleAgent] = []

        self.battlespace.all_agents.append(agent)

    def get_observation_space(self,
                              is_pursuer: bool) -> gym.spaces.Dict:
        """
        Initializes the observation space for the agent
        in a form of a dictionary
        Dictionary contains the following keys:

        - observations: The state vector of the agent
        - action_mask: The action mask for the agent

        Observations are in the form of a Box space  which contain
        the following information:
            Relative position is defined as (target_pos - agent_pos)
            - 0:x
            - 1:y
            - 2:z
            - 3:roll: The roll of the agent
            - 4:pitch: The pitch of the agent
            - 5:yaw: The yaw of the agent
            - 6:speed: The speed of the agent
        """
        if is_pursuer:
            obs_config = self.pursuer_state_limits
        else:
            obs_config = self.evader_state_limits

        high_obs, low_obs = self.map_config(obs_config)
        obs_bounds = gym.spaces.Box(low=np.array(
            low_obs), high=np.array(high_obs), dtype=np.float32)
        # Define an action mask space that matches the shape of your discrete action space.
        # For MultiDiscrete, you might use MultiBinary with the same nvec.
        total_actions = int(self.action_space.nvec.sum())
        mask_space = gym.spaces.MultiBinary(total_actions)
        observation = gym.spaces.Dict({
            "action_mask": mask_space,
            "observations": obs_bounds
        })

        return observation

    def get_discrete_action_space(self,
                                  is_pursuer: bool = True) -> gym.spaces.MultiDiscrete:
        """
        Args:
            None

        Returns:
            The action space for the environment.

        Initializes the action space for the environment
        For this environment the action space will be a discrete space
        where the aircraft can send commands in the form of u_roll, u_pitch, u_yaw, u_speed

        action_space:
            [
                u_roll_idx:[0, 1, 2, ...n],
                u_cmd_idx :[0, 1, 2, ...n],
                u_yaw_idx :[0, 1, 2, ...n],
                u_speed_idx:[0, 1, 2, ...n]
            ]

        This is mapped to the continous action space
        To get the actual commands use the discrete_to_continuous_action method
        """
        if is_pursuer:
            control_config = self.pursuer_control_limits
        else:
            control_config = self.evader_control_limits

        continous_action_space: gym.spaces.Box = self.get_continous_action_space(
            control_limits=control_config
        )
        roll_idx: int = 0
        pitch_idx: int = 1
        yaw_idx: int = 2
        vel_idx: int = 3
        self.roll_commands: np.array = np.arange(
            continous_action_space.low[roll_idx], continous_action_space.high[roll_idx],
            np.deg2rad(5))
        self.pitch_commands: np.array = np.arange(
            continous_action_space.low[pitch_idx], continous_action_space.high[pitch_idx],
            1)
        self.yaw_commands: np.array = np.arange(
            continous_action_space.low[yaw_idx], continous_action_space.high[yaw_idx],
            1)
        self.airspeed_commands: np.array = np.arange(
            continous_action_space.low[vel_idx], continous_action_space.high[vel_idx],
            1)

        action_space = gym.spaces.MultiDiscrete(
            [len(self.roll_commands), len(self.pitch_commands),
             len(self.yaw_commands), len(self.airspeed_commands)])

        return action_space

    def map_config(self, control_config: Dict[str, Any]) -> Tuple[List, List]:
        """
        Returns a tuple of the high and low values from the configuration
        file
        """
        high = []
        low = []
        for k, v in control_config.items():
            for inner_k, inner_v in v.items():
                if 'max' in inner_k:
                    high.append(inner_v)
                elif 'min' in inner_k:
                    low.append(inner_v)

        return high, low

    def get_continous_action_space(self,
                                   control_limits: Dict[str, Any]) -> gym.spaces.Box:
        """
        Initializes the action space for the environment
        """
        # agent: Agent = self.get_controlled_agents[0]
        high, low = self.map_config(control_limits)

        return gym.spaces.Box(low=np.array(low),
                              high=np.array(high),
                              dtype=np.float32)

    def get_action_mask(self, agent: SimpleAgent) -> np.ndarray:
        """
        Returns the action mask for the agent
        This will be used to mask out the actions that are not allowed
        From this abstract class we provide methods to make sure 
        agent is not allowed to go out of bounds
        You can call override this method if you want
        """
        action_mask = np.zeros(self.action_space.nvec.sum(), dtype=int)

        # Get agent information

        # Mask pitch commands based on how close it is to z bounds

        # Mask roll/yaw commands based on how close it is to x/y bounds

        return action_mask


class EngageEnv(AbstracKinematicEnv):
    def __init__(self, config: Optional[Dict]) -> None:
        super(EngageEnv, self).__init__(config=config)

        self.action_space = self.init_action_space()
        self.observation_space = self.init_observation_space()

    def init_action_space(self) -> gym.spaces.MultiDiscrete:
        return self.get_discrete_action_space(is_pursuer=True)

    def init_observation_space(self) -> gym.spaces.Dict:
        return self.get_observation_space(is_pursuer=True)

    def step(self, action: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """
        """
        obs, reward, done, info = {}, {}, {}, {}
        # for agent_id, act in action.items():
        #     agent: SimpleAgent = self.battlespace.get_agent_by_id(agent_id)
        #     obs[agent_id], reward[agent_id], done[agent_id], info[agent_id] = agent.step(
        #         action=act)

        return obs, reward, done, info


class AvoidEnv(AbstracKinematicEnv):
    def __init__agents(self):
        return super().__init__agents()
