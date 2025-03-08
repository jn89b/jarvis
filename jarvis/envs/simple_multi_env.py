import yaml
import gymnasium as gym
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from ray.rllib.env import MultiAgentEnv

from jarvis.envs.simple_agent import (
    SimpleAgent, Pursuer, Evader, PlaneKinematicModel)
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector
from jarvis.envs.tokens import KinematicIndex
from jarvis.algos.pro_nav import ProNavV2
import itertools

# abstract methods
from abc import ABC, abstractmethod


class AbstracKinematicEnv(gym.Env, ABC):
    """

    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        # Call the next __init__ in the MRO (likely gym.Env.__init__)
        super().__init__(**kwargs)
        # Optionally store the config or initialize later.
        if config is not None:
            self.init_config(config)

        self.config = None
        self.agent_config = None
        self.spawn_config = None
        self.simulation_config = None
        self.current_step = 0
        self.sim_end_time = None
        self.dt = None
        self.max_steps = None
        self.sim_frequency = None
        self.old_action = None
        self.ctrl_time = None
        self.ctrl_time_index = None
        self.ctrl_counter = None
        self.roll_commands = None
        self.pitch_commands = None
        self.yaw_commands = None
        self.airspeed_commands = None
        self.pursuer_control_limits = None
        self.pursuer_state_limits = None
        self.evader_control_limits = None
        self.evader_state_limits = None
        self.battlespace = None
        self.agents = None

    def init_config(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = config.get("agents")
        self.spawn_config: Dict[str, Any] = config.get("spawning")
        self.simulation_config: Dict[str, Any] = config.get("simulation")

        if self.simulation_config is None:
            raise ValueError(
                "Simulation configuration is required refer to simple_env_config.yaml")

        if self.agent_config is None:
            raise ValueError(
                "Agent configuration is required refer to simple_env_config.yaml")

        if self.spawn_config is None:
            raise ValueError(
                "Spawning configuration is required refer to simple_env_config.yaml")

        self.current_step: int = 0
        self.sim_end_time: int = self.simulation_config.get("end_time")
        self.dt: float = self.simulation_config.get("dt")
        self.max_steps: int = int(self.sim_end_time / self.dt)
        self.sim_frequency: int = int(1/self.dt)

        # this is used to control how often the agent can update its high level control
        self.old_action: np.array = None
        self.ctrl_time: float = 0.1  # control time in seconds
        self.ctrl_time_index = int(self.ctrl_time * self.sim_frequency)
        self.ctrl_counter: int = 0  # control counter

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

    def remove_all_agents(self) -> None:
        """
        Removes all agents from the environment
        """
        self.battlespace.clear_agents()
        self.agents = []

    def insert_agent(self, agent: SimpleAgent) -> None:
        if agent.is_controlled:
            self.agents.append(agent.agent_id)

        self.battlespace.all_agents.append(agent)

    def get_specific_agent(self, agent_id: int) -> SimpleAgent:
        """
        Returns a specific agent in the environment
        """
        for agent in self.battlespace.all_agents:
            if agent.agent_id == agent_id:
                return agent

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

    def get_evader_agents(self) -> List[SimpleAgent]:
        """
        Returns all the evader agents in the environment
        """
        evader_agents = []
        for agent in self.battlespace.all_agents:
            if not agent.is_pursuer:
                evader_agents.append(agent)

        return evader_agents

    def get_pursuer_agents(self) -> List[SimpleAgent]:
        """
        Returns all the pursuer agents in the environment
        """
        pursuer_agents = []
        for agent in self.battlespace.all_agents:
            if agent.is_pursuer:
                pursuer_agents.append(agent)

        return pursuer_agents

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
            # control_limits = self.evader_control_limits
            rand_x = np.random.uniform(-10, 10)
            rand_y = np.random.uniform(-10, 10)
            rand_z = np.random.uniform(
                state_limits['z']['min']-20, state_limits['z']['max']+20)

            rand_phi = np.random.uniform(state_limits['phi']['min'],
                                         state_limits['phi']['max'])
            rand_theta = np.random.uniform(state_limits['theta']['min'],
                                           state_limits['theta']['max'])
            rand_psi = np.random.uniform(-np.pi, np.pi)

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

    def build_observation_space(self,
                                is_pursuer: bool,
                                num_actions: int = None) -> gym.spaces.Dict:
        """
        Initializes the observation space for the agent
        in a form of a dictionary
        Dictionary contains the following keys:

        - observations: The state vector of the agent
        - action_mask: The action mask for the agent
        - num_actions: The number of actions the agent can take

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
        if num_actions is None:
            total_actions = self.action_space.nvec.sum()
        else:
            total_actions = num_actions

        mask_space = gym.spaces.MultiBinary(total_actions)
        observation = gym.spaces.Dict({
            "action_mask": mask_space,
            "observations": obs_bounds
        })

        return observation

    def discrete_to_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Args:
            action (np.ndarray): The discrete action to convert to continous action
            refer to the get_discrete_action_space method for more information

        Returns:
            np.ndarray: The continous action space in the form of [roll, pitch, yaw, vel_cmd]

        Converts the discrete action space to a continous action space
        """
        if self.roll_commands is None or self.pitch_commands is None or self.yaw_commands is None or self.airspeed_commands is None:
            raise ValueError(
                "The roll, pitch, yaw, and airspeed commands are not initialized")

        roll_idx: int = action[0]
        pitch_idx: int = action[1]
        yaw_idx: int = action[2]
        vel_idx: int = action[3]

        roll_cmd: float = self.roll_commands[roll_idx]
        pitch_cmd: float = self.pitch_commands[pitch_idx]
        yaw_cmd: float = self.yaw_commands[yaw_idx]
        vel_cmd: float = self.airspeed_commands[vel_idx]

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, vel_cmd], dtype=np.float32)

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
            np.deg2rad(1))
        self.yaw_commands: np.array = np.arange(
            continous_action_space.low[yaw_idx], continous_action_space.high[yaw_idx],
            np.deg2rad(1))
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

    def compute_ascent_descent_rate(self, current_vel: float,
                                    pitch_cmd: float) -> float:
        return current_vel * np.sin(pitch_cmd)

    def mask_pitch_commands(self, agent: SimpleAgent, pitch_mask: np.ndarray,
                            z_bounds: Dict[str, List[float]],
                            projection_time: float = 0.5) -> np.ndarray:
        """
        Args:
            agent (SimpleAgent): The agent to mask the pitch commands for
            action_mask (np.ndarray): The action mask to update
            z_bounds (List[float]): The bounds of the agent
            projection_time (float): The time to project the agent forward
        Returns:
            np.ndarray: The updated action mask

        This method masks the pitch commands based on how close the agent is to the z bounds
        We do this by projecting the agent forward in time and checking if the agent will
        hit the z bounds based on current velocity and the range of pitch commands
        """
        z_position = agent.state_vector.z
        v_current = agent.state_vector.speed
        for i, pitch_cmd in enumerate(self.pitch_commands):
            ascent_descent_rate = self.compute_ascent_descent_rate(
                current_vel=v_current, pitch_cmd=pitch_cmd)
            projected_z = z_position + (ascent_descent_rate * projection_time)
            if projected_z > z_bounds['max'] or projected_z < z_bounds['min']:
                pitch_mask[i] = 0

        return pitch_mask

    def mask_psi_commands(self, agent: SimpleAgent, yaw_mask: np.ndarray,
                          x_bounds: Dict[str, List[float]], y_bounds: Dict[str, List[float]],
                          projection_time: float = 0.5,
                          consider_target: bool = False) -> None:
        """
        Args:
            agent (SimpleAgent): The agent to mask the yaw commands for
            psi_mask (np.ndarray): The action mask to update
            consider_target (bool): True if we are considering the target, False otherwise

        Returns:
            np.ndarray: The updated action mask

        This is done by projecting the agent forward in time based on the yaw commands
        from the action space with a simple Euler integration
        #TODO: Can call out RK45 here instead of Euler but it works

        if the agent is close to the x or y bounds
        we mask the yaw commands that will take the agent out of bounds

        In addition if the user wants to consider the target we can do that as well
        This is useful to mask possible commands that will either collide with the the target

        """

        current_speed: float = agent.state_vector.speed
        current_state: float = agent.simple_model.state_info
        for i, yaw_cmd in enumerate(self.yaw_commands):
            u: np.ndarray = np.array([0,
                                      0,
                                      yaw_cmd,
                                      current_speed])
            next_state: np.array = agent.simple_model.rk45(
                x=current_state, u=u, dt=projection_time)
            x_proj: float = next_state[KinematicIndex.X.value]
            y_proj: float = next_state[KinematicIndex.Y.value]
            print("x_proj", x_proj)

            if x_proj > x_bounds['max'] or x_proj < x_bounds['min']:
                yaw_mask[i] = 0
            if y_proj > y_bounds['max'] or y_proj < y_bounds['min']:
                yaw_mask[i] = 0

        return yaw_mask

    def unwrap_action_mask(self, action_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits the action mask into roll, pitch, yaw, and velocity
        masks
        """
        unwrapped_mask: Dict[str, np.array] = {
            'roll': action_mask[:len(self.roll_commands)],
            'pitch': action_mask[len(self.roll_commands):len(self.roll_commands) + len(self.pitch_commands)],
            'yaw': action_mask[len(self.roll_commands) + len(self.pitch_commands):len(self.roll_commands) + len(self.pitch_commands) + len(self.yaw_commands)],
            'vel': action_mask[len(self.roll_commands) + len(self.pitch_commands) + len(self.yaw_commands):]
        }

        return unwrapped_mask

    def wrap_action_mask(self, unwrapped_mask: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Wraps the action mask into a single array
        """
        action_mask = np.concatenate([unwrapped_mask['roll'],
                                      unwrapped_mask['pitch'],
                                      unwrapped_mask['yaw'],
                                      unwrapped_mask['vel']])

        return np.array(action_mask, dtype=np.int8)

    def get_action_mask(self, agent: SimpleAgent,
                        action_space_sum: int = None) -> np.ndarray:
        """
        Returns the action mask for the agent
        This will be used to mask out the actions that are not allowed
        From this abstract class we provide methods to make sure
        agent is not allowed to go out of bounds
        You can call override this method if you want
        """
        if agent.is_pursuer:
            x_bounds: List[float] = self.pursuer_state_limits['x']
            y_bounds: List[float] = self.pursuer_state_limits['y']
            z_bounds: List[float] = self.pursuer_state_limits['z']
        else:
            x_bounds: List[float] = self.evader_state_limits['x']
            y_bounds: List[float] = self.evader_state_limits['y']
            z_bounds: List[float] = self.evader_state_limits['z']

        # Mask roll/yaw commands based on how close it is to x/y bounds
        roll_mask: np.array = np.ones_like(self.roll_commands, dtype=np.int8)
        pitch_mask: np.array = np.ones_like(self.pitch_commands, dtype=np.int8)
        yaw_mask: np.array = np.ones_like(self.yaw_commands, dtype=np.int8)
        vel_mask: np.array = np.ones_like(
            self.airspeed_commands, dtype=np.int8)

        pitch_mask: np.array = self.mask_pitch_commands(agent=agent,
                                                        pitch_mask=pitch_mask,
                                                        z_bounds=z_bounds)

        yaw_mask: np.array = self.mask_psi_commands(agent=agent,
                                                    yaw_mask=yaw_mask,
                                                    x_bounds=x_bounds,
                                                    y_bounds=y_bounds)

        full_mask = np.concatenate([roll_mask, pitch_mask, yaw_mask, vel_mask])

        if action_space_sum is not None:
            assert full_mask.size == action_space_sum
        else:
            assert full_mask.size == self.action_space.nvec.sum()

        return full_mask

    def simulate(self, action_dict: np.ndarray, use_multi: bool = False) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step()

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self.battlespace.clear_agents()
        self.__init_battlespace()
        self.__init__agents()
        self.current_step = 0

    def observe(self, agent: SimpleAgent,
                total_actions: int = None) -> Dict[str, np.ndarray]:
        """
        Args:
            None
        Returns:
            A dictionary containing the observations for the agent
            in the environment. The dictionary contains the following keys:
            - observations: The state vector of the agent
            - action_mask: The action mask for the agent

        Observations are in the form of a Box space  which contain
        [dx, dy, dz, roll, pitch, yaw, speed]

        Refer to build_observation_space for the observation space definition.
        """
        obs = [agent.state_vector.x,
               agent.state_vector.y,
               agent.state_vector.z,
               agent.state_vector.roll_rad,
               agent.state_vector.pitch_rad,
               agent.state_vector.yaw_rad,
               agent.state_vector.speed,
               agent.state_vector.vx,
               agent.state_vector.vy,
               agent.state_vector.vz]

        obs = np.array(obs, dtype=np.float32)
        # obs_space: Dict[str, gym.spaces.Box] = self.build_observation_space(
        #     is_pursuer=agent.is_pursuer)
        # low = obs_space['observations'].low
        # high = obs_space['observations'].high
        # clip the psi
        # obs[KinematicIndex.YAW] = np.clip(obs[KinematicIndex.YAW.])
        # if np.any(obs < low) or np.any(obs > high):
        #     # print the one out of bounds
        #     for i, (obs_val, low_val, high_val) in enumerate(zip(obs, low, high)):
        #         if obs_val < low_val or obs_val > high_val:
        #             raise ValueError("Observation out of bounds",
        #                              f"Observation {i} out of bounds: {obs_val} not in [{low_val}, {high_val}]")

        # make sure obs is np.float32
        obs = np.array(obs, dtype=np.float32)
        action_mask: np.ndarray = self.get_action_mask(agent=agent,
                                                       action_space_sum=total_actions)
        return {'observations': obs, 'action_mask': action_mask}


class EngageEnv(AbstracKinematicEnv):
    def __init__(self, config: Optional[Dict]) -> None:
        super(EngageEnv, self).__init__(config=config)
        self.init_config(config)
        self.relative_observations: Dict[str,
                                         Any] = self.agent_config['relative_state_observations']
        self.action_space = self.init_action_space()
        self.observation_space = self.init_observation_space()
        self.__init_target()
        self.agent_interaction: Dict[str,
                                     Any] = self.agent_config['interaction']
        self.terminal_reward: float = 100.0
        # used for reward calculation
        self.old_distance_from_target: float = self.target.distance_3D(
            self.get_controlled_agents[0].state_vector
        )
        self.old_dot_product: float = self.target.dot_product_2D(
            self.get_controlled_agents[0].state_vector
        )
        self.pn = ProNavV2()

    def init_action_space(self) -> gym.spaces.MultiDiscrete:
        return self.get_discrete_action_space(is_pursuer=True)

    def init_observation_space(self) -> gym.spaces.Dict:
        """
        We need to include relative position of the agent to the target
        """
        relative_positions: Dict[str,
                                 Any] = self.relative_observations['position']
        print("relative_positions: ", relative_positions)
        low_rel_pos: List[float] = [relative_positions['x']['low'],
                                    relative_positions['y']['low'],
                                    relative_positions['z']['low']]

        high_rel_pos: List[float] = [relative_positions['x']['high'],
                                     relative_positions['y']['high'],
                                     relative_positions['z']['high']]

        observation_space = self.build_observation_space(is_pursuer=True)
        obs_bounds_low: np.array = observation_space['observations'].low
        obs_bounds_high: np.array = observation_space['observations'].high

        obs_bounds_low = np.concatenate([obs_bounds_low, low_rel_pos])
        obs_bounds_high = np.concatenate([obs_bounds_high, high_rel_pos])

        action_mask: np.array = observation_space['action_mask']

        observation = gym.spaces.Dict({
            "action_mask": action_mask,
            "observations": gym.spaces.Box(low=obs_bounds_low,
                                           high=obs_bounds_high, dtype=np.float32)
        })

        return observation

    def __init_target(self) -> None:
        self.target_config: Dict[str, Any] = self.spawn_config['target']
        if self.target_config is None:
            raise ValueError(
                "Target configuration is required refer to simple_env_config.yaml")

        self.randomize_target: bool = self.target_config['randomize']

        if self.get_controlled_agents[0].is_pursuer:
            state_limits: Dict[str, Dict[str, float]
                               ] = self.pursuer_state_limits
        else:
            state_limits: Dict[str, Dict[str, float]
                               ] = self.evader_state_limits

        state_limits: Dict[str, Dict[str, float]] = self.pursuer_state_limits
        if self.randomize_target:
            min_radius: float = self.target_config['spawn_radius_from_agent']['min']
            max_radius: float = self.target_config['spawn_radius_from_agent']['max']

            # random heading
            rand_heading = np.random.uniform(0, 2 * np.pi)

            # get the agent position
            agent: SimpleAgent = self.get_controlled_agents[0]

            target_x = agent.state_vector.x + \
                np.random.uniform(min_radius, max_radius)*np.cos(rand_heading)

            target_y = agent.state_vector.y + \
                np.random.uniform(min_radius, max_radius)*np.sin(rand_heading)

            # so this is confusing but the way we spawn the target
            # is going to be based on
            target_z = np.random.uniform(state_limits['z']['min'] + 15,
                                         state_limits['z']['max'] - 15)

            self.target = StateVector(x=target_x, y=target_y, z=target_z,
                                      roll_rad=0, pitch_rad=0, yaw_rad=0, speed=0)

        else:
            x: float = self.target_config['position']['x']
            y: float = self.target_config['position']['y']
            z: float = self.target_config['position']['z']
            self.target = StateVector(x=x,
                                      y=y,
                                      z=z,
                                      roll_rad=0, pitch_rad=0, yaw_rad=0, speed=0)

    def compute_intermediate_reward(self, agent: SimpleAgent) -> float:
        """
        Args:
            agent (SimpleAgent): The agent to compute the reward for
        Returns:
            float: The intermediate reward for the agent

        Computes the intermediate reward for the agent
        For this case we want to maximize us closing in on the target

        So if we computed old_distance - new_distance
        We want it where this value is positive ie: 5 - 3 = 2

        If we are moving away from the target we want to penalize the agent
        5 - 7 = -2

        """
        distance_to_target: float = -self.target.distance_3D(
            agent.state_vector)
        alpha = 0.01
        return alpha * distance_to_target

    def is_close_to_target(self, agent: SimpleAgent) -> bool:
        """
        Args:
            agent (SimpleAgent): The agent to check if it is close to the target
        Returns:
            bool: True if the agent is close to the target, False otherwise

        """
        capture_radius: float = self.agent_interaction['capture_radius']

        distance: float = self.target.distance_3D(agent.state_vector)
        if distance <= capture_radius:
            return True

        return False

    def update_pitch_mask(self, agent: SimpleAgent,
                          pitch_mask: np.array,
                          buffer_height_diff_m: float = 10) -> np.ndarray:
        """
        Args:
            agent (SimpleAgent): The agent to update the pitch mask for
            pitch_mask (np.array): The pitch mask to update
            buffer_height_diff_m (float): The buffer height difference in meters

        Returns:
            np.ndarray: The updated pitch mask

        This method updates the pitch mask based on the altitude difference
        between the target and the agent

        If target is ABOVE us we want to mask the pitch commands
        that will make us go DOWN

        If target is BELOW us we want to mask the pitch commands
        that will make us go UP
        """
        dz = self.target.z - agent.state_vector.z
        if dz > buffer_height_diff_m:
            # target is above us
            # mask the pitch commands that will make us go down
            # so begining of the pitch commands to the middle
            pitch_mask[:len(self.pitch_commands)//2 + 2] = 0
        elif dz < -buffer_height_diff_m:
            # target is below us
            # mask the pitch commands that will make us go up
            # so middle to the end of the pitch commands to the end
            pitch_mask[len(self.pitch_commands)//2 + 2:] = 0
        else:
            # target is at the same height as us
            # we can use all the pitch commands
            # just mask the outer pitch commands
            pitch_mask[:2] = 0
            pitch_mask[-2:] = 0

        return pitch_mask

    def update_yaw_mask(self, agent: SimpleAgent,
                        yaw_mask: np.array,
                        buffer_angle_diff: float = 0.1) -> np.ndarray:
        """
        Args:
            agent (SimpleAgent): The agent to update the yaw mask for
            yaw_mask (np.array): The yaw mask to update
            buffer_angle_diff (float): The buffer angle difference

        Returns:
            np.ndarray: The updated yaw mask


        Updates the yaw mask based on the angle difference between the target
        and the agent by using the dot product of the target and the agent

        If the target is to the right of us we want to mask the yaw commands
        that will make us go left

        If the target is to the left of us we want to mask the yaw commands
        that will make us go right

        KEEP in mind this is based on your frame 
        if you are using ENU/NEU notation to turn right 
        you neeed a negative yaw command to turn clockwise

        Remember yaw commands
        """
        # dot_product: float = self.target.dot_product_2D(agent.state_vector)
        dx = self.target.x - agent.state_vector.x
        dy = self.target.y - agent.state_vector.y
        # print("dx and dy", dx, dy)
        los: float = np.arctan2(dy, dx)
        delta_yaw: float = los - agent.state_vector.yaw_rad
        # print("delta_yaw: ", np.rad2deg(delta_yaw))
        # wrap between -pi and pi
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        buffer_angle: float = np.deg2rad(15)

        # if delta_yaw > buffer_angle:
        #     pass
        #     # Need to turn left so CCW turn or positive yaw commands
        #     # since yaw commands are global we need to mask super far yaw commands
        #     # print("To the left Desired Yaw: ", np.rad2deg(delta_yaw))
        #     # grab the index of the yaw commands that are close to the desired yaw
        # elif delta_yaw < -buffer_angle:
        #     # print("To the right Desired Yaw: ", np.rad2deg(delta_yaw))
        #     index_desired = np.abs(self.yaw_commands - delta_yaw).argmin()

        index_desired = np.abs(self.yaw_commands - delta_yaw).argmin()

        # mask values that are not close to the desired yaw
        yaw_mask[:index_desired - 15] = 0
        yaw_mask[index_desired + 15:] = 0

        return yaw_mask

    def observe(self, agent) -> Dict[str, np.ndarray]:
        observation: Dict[str, np.ndarray] = super().observe(agent)
        obs: np.ndarray = observation['observations']

        action_mask: np.array = observation['action_mask']
        unpacked_mask: Dict[str, np.ndarray] = self.unwrap_action_mask(
            action_mask)

        pitch_mask: np.array = unpacked_mask['pitch']
        yaw_mask: np.array = unpacked_mask['yaw']
        unpacked_mask['pitch'] = self.update_pitch_mask(agent=agent,
                                                        pitch_mask=pitch_mask)
        unpacked_mask['yaw'] = self.update_yaw_mask(
            agent=agent, yaw_mask=yaw_mask)
        # we want to update this mask based on the altitude difference between
        # the target and the agent
        action_mask = self.wrap_action_mask(unpacked_mask)
        target_obs: np.ndarray = self.target.array
        relative_pos: np.ndarray = target_obs[:3] - obs[:3]
        # clip the relative position
        low = self.relative_observations['position']['x']['low']
        high = self.relative_observations['position']['x']['high']
        relative_pos = np.clip(relative_pos, low, high)

        obs = np.concatenate([obs, relative_pos])
        obs = np.array(obs, dtype=np.float32)
        observation['observations'] = obs
        observation['action_mask'] = action_mask
        return observation

    def step(self, action: np.array) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """
        Keep in mind the yaw command is in NEU notation
        """
        reward: float = 0.0
        truncated: bool = False
        terminated: bool = False
        info = {}

        # continous_action: np.array = self.discrete_to_continuous_action(action)
        # self.simulate(action_dict=continous_action, use_multi=False)
        # agent: SimpleAgent = self.get_controlled_agents[0]

        if self.ctrl_counter % self.ctrl_time_index == 0 and self.ctrl_counter != 0 \
                or self.old_action is None:
            action: np.ndarray = self.discrete_to_continuous_action(action)
            self.simulate(action, use_multi=False)
            self.ctrl_counter = 0        # self.config: Dict[str, Any] = config
        # self.agent_config: Dict[str, Any] = config.get("agents")
        # self.spawn_config: Dict[str, Any] = config.get("spawning")
        # self.simulation_config: Dict[str, Any] = config.get("simulation")

        # if self.simulation_config is None:
        #     raise ValueError(
        #         "Simulation configuration is required refer to simple_env_config.yaml")

        # if self.agent_config is None:
        #     raise ValueError(
        #         "Agent configuration is required refer to simple_env_config.yaml")

        # if self.spawn_config is None:
        #     raise ValueError(
        #         "Spawning configuration is required refer to simple_env_config.yaml")

        # self.current_step: int = 0
        # self.sim_end_time: int = self.simulation_config.get("end_time")
        # self.dt: float = self.simulation_config.get("dt")
        # self.max_steps: int = int(self.sim_end_time / self.dt)
        # self.sim_frequency: int = int(1/self.dt)

        # # this is used to control how often the agent can update its high level control
        # self.old_action: np.array = None
        # self.ctrl_time: float = 0.1  # control time in seconds
        # self.ctrl_time_index = int(self.ctrl_time * self.sim_frequency)
        # self.ctrl_counter: int = 0  # control counter

        # # these methods will be implemented by the user
        # self.roll_commands: np.array = None
        # self.pitch_commands: np.array = None
        # self.yaw_commands: np.array = None
        # self.airspeed_commands: np.array = None

        # self.pursuer_control_limits, self.pursuer_state_limits = self.load_limit_config(
        #     is_pursuer=True)
        # self.evader_control_limits, self.evader_state_limits = self.load_limit_config(
        #     is_pursuer=False)

        # self.__init_battlespace()
        # assert self.battlespace is not None
        # self.__init__agents()
        # self.agents: List[int] = [
        #     agent.agent_id for agent in self.get_controlled_agents]
            self.old_action: np.ndarray = action
        else:
            self.ctrl_counter += 1
            self.simulate(self.old_action, use_multi=False)

        agent: SimpleAgent = self.get_controlled_agents[0]
        observation: Dict[str, np.ndarray] = self.observe(agent=agent)
        reward: float = self.compute_intermediate_reward(agent=agent)
        distance: float = self.target.distance_3D(agent.state_vector)
        if self.current_step >= self.max_steps:
            print("ran out of time", distance)
            self.terminal_reward = -self.terminal_reward
            terminated: bool = True
        elif self.is_close_to_target(agent=self.get_controlled_agents[0]):
            print("Captured", distance)
            reward = self.terminal_reward
            terminated: bool = True
        elif agent.crashed:
            print("Agent Crashed")
            reward = -self.terminal_reward
            terminated: bool = True

        # for agent in self.get_controlled_agents:
        #     distance: float = self.target.distance_3D(agent.state_vector)
        #     info[agent.agent_id] = {
        #         'crashed': agent.crashed,
        #         'reward': reward,
        #         'distance_to_goal': distance
        #     }
        # for being slow
        reward -= 0.01
        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.__init_target()

        observation = self.observe(agent=self.get_controlled_agents[0])
        info = {}
        return observation, info


class AvoidEnv(AbstracKinematicEnv):
    def __init__agents(self):
        return super().__init__agents()


class PursuerEvaderEnv(MultiAgentEnv, AbstracKinematicEnv):
    """
    Evader where must avoid being captured by the pursuers must 
    get to the target location 
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        # Use super() so that the MRO calls MultiAgentEnv.__init__ then AbstracKinematicEnv.__init__
       # Now call the initializer for your kinematic environment with config.
        MultiAgentEnv.__init__(self)
        AbstracKinematicEnv.__init__(self, config=config, **kwargs)
        self.init_config(config)
        # super(PursuerEvaderEnv, self).__init__(config=config)
        self.interaction_config: Dict[str,
                                      Any] = self.agent_config['interaction']
        self.relative_state_observations: Dict[str,
                                               Any] = self.agent_config['relative_state_observations']
        self.action_spaces = self.init_action_space()
        self.observation_spaces = self.init_observation_space()
        self.current_step: int = 0
        self.possible_agents: List[int] = self.agents
        self.agent_cycle = itertools.cycle(self.possible_agents)
        self.current_agent: int = next(self.agent_cycle)
        self.terminal_reward: float = 100.0
        self.all_done_step: int = 0
        self.pro_nav = ProNavV2()
        self.use_pn: bool = True

    def init_action_space(self) -> Dict[str, gym.spaces.Dict]:
        """
        Returns the action space for the pursuers and evaders
        """
        # self.action_spaces: Dict[str, gym.spaces.Dict] = {}
        # for agent in self.get_controlled_agents:
        #     action_space: gym.spaces.Dict = self.get_discrete_action_space(
        #         is_pursuer=agent.is_pursuer
        #     )
        #     self.action_spaces[agent.agent_id] = action_space

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
            x_low: float = relative_positions['x']['low']
            y_low: float = relative_positions['y']['low']
            z_low: float = relative_positions['z']['low']

            x_high: float = relative_positions['x']['high']
            y_high: float = relative_positions['y']['high']
            z_high: float = relative_positions['z']['high']

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

                obs_relative_low.extend([x_low, y_low, z_low])
                obs_relative_high.extend([x_high, y_high, z_high])

            obs_low = np.concatenate([obs_low, obs_relative_low])
            obs_high = np.concatenate([obs_high, obs_relative_high])
            obs = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            observation_space['observations'] = obs

            self.observation_spaces[agent.agent_id] = observation_space

        return self.observation_spaces

    def observe(self, agent: SimpleAgent) -> Dict[str, np.ndarray]:
        """
        Args:
            agent (SimpleAgent): The agent to observe

        Returns:
            Dict[str, np.ndarray]: The observation for the agent

        Observation is in the form of a dictionary containing the following keys:
        - observations: The state vector of the agent
        - action_mask: The action mask for the agent

        The observation is in the form of a Box space

        which contains the following information:
            - x: The x position of the agent
            - y: The y position of the agent
            - z: The z position of the agent
            - roll: The roll of the agent
            - pitch: The pitch of the agent
            - yaw: The yaw of the agent
            - speed: The speed of the agent
            - vx: The x velocity of the agent
            - vy: The y velocity of the agent
            - vz: The z velocity of the agent

            For n other agents in the environment we include the relative positions
            - relative_x: The relative x position of the agent to the other agents
            - relative_y: The relative y position of the agent to the other agents
            - relative_z: The relative z position of the agent to the other agents

        Relative is defined as (agent_pos - other_agent_pos)

        """
        # action_sum = self.action_spaces[agent.agent_id].nvec.sum()
        action_sum = self.action_spaces[agent.agent_id]["action"].nvec.sum()
        observation: Dict[str, np.ndarray] = super().observe(agent, action_sum)
        obs: np.ndarray = observation['observations']
        action_mask: np.ndarray = observation['action_mask']

        overall_relative_pos: List[float] = []
        for other_agent in self.get_controlled_agents:
            if agent.agent_id == other_agent.agent_id:
                continue

            if agent.is_pursuer == other_agent.is_pursuer:
                continue

            relative_pos: np.ndarray = agent.state_vector.array - \
                other_agent.state_vector.array
            relative_pos = relative_pos[:3]
            overall_relative_pos.extend(relative_pos)

        obs = np.concatenate([obs, overall_relative_pos])
        observation['observations'] = obs

        return observation

    def is_caught(self, pursuer: SimpleAgent, evader: SimpleAgent) -> bool:
        """
        Args:
            pursuer (SimpleAgent): The pursuer agent
            evader (SimpleAgent): The evader agent

        Returns:
            bool: True if the evader is caught by the pursuer, False otherwise

        The evader is caught if the pursuer is within the capture radius
        of the evader
        """
        capture_radius: float = self.interaction_config['capture_radius']
        distance: float = pursuer.state_vector.distance_3D(evader.state_vector)
        if distance <= capture_radius:
            return True

        return False

    def compute_pursuer_reward(self, pursuer: Pursuer, evader: Evader) -> float:
        """
        Compute the distance between the pursuer and the evader
        Reward for closing the distance between the pursuer and the evader
        """
        old_distance: float = pursuer.old_distance_from_evader
        distance_from_evader: float = pursuer.state_vector.distance_3D(
            evader.state_vector)
        # ie 5 - 3 = 2
        delta_distance = old_distance - distance_from_evader
        alpha: float = 1.0
        pursuer.old_distance_from_evader = distance_from_evader

        return alpha * delta_distance

    def compute_evader_reward(self, pursuer: Pursuer, evader: Evader) -> float:
        """
        Compute the distance between the pursuer and the evader
        Reward for maximizing the distance between the pursuer and the evader
        """
        old_distance: float = evader.old_distance_from_evader

        distance_from_evader: float = evader.state_vector.distance_3D(
            pursuer.state_vector)

        # ie 5 - 3 = 2 this means the pursuer is closing in on the evader
        delta_distance = distance_from_evader - old_distance
        alpha: float = 1.0
        # evader.old_distance_from_evader = distance_from_evader

        return alpha * delta_distance

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        """
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        rewards = {agent: 0.0 for agent in self.agents}
        infos = {}
        observations = {}
        agent: SimpleAgent = self.get_specific_agent(self.current_agent)

        action: np.array = self.discrete_to_continuous_action(
            action_dict[agent.agent_id]['action'])

        if agent.is_pursuer and self.use_pn:
            state_vector: StateVector = agent.state_vector
            current_pos: np.array = np.array([
                state_vector.x,
                state_vector.y,
                state_vector.z,

            ])
            evader: Evader = self.get_evader_agents()[0]
            relative_pos: np.array = np.array([
                state_vector.x - evader.state_vector.x,
                state_vector.y - evader.state_vector.y,
                state_vector.z - evader.state_vector.z
            ])
            relative_vel = state_vector.speed - evader.state_vector.speed
            action = self.pro_nav.predict(
                current_pos=current_pos,
                relative_pos=relative_pos,
                relative_vel=relative_vel,
                current_speed=state_vector.speed)

            # let's use pn instead of the discrete action

        command_action: Dict[str, np.array] = {agent.agent_id: action}
        self.simulate(command_action, use_multi=True)

        observations = self.observe(agent=agent)
        evaders: List[SimpleAgent] = self.get_evader_agents()
        evader = evaders[0]

        for agent in self.get_controlled_agents:
            # rewards for the pursuers
            if agent.is_pursuer:
                if self.is_caught(pursuer=agent, evader=evader) or evader.crashed:
                    for pursuer in self.get_pursuer_agents():
                        rewards[pursuer.agent_id] = self.terminal_reward
                        rewards[evader.agent_id] = -self.terminal_reward
                        terminateds['__all__'] = True
                        break
                else:
                    # compute the intermediate reward
                    rewards[agent.agent_id] = self.compute_pursuer_reward(
                        pursuer=agent, evader=evader)
                    rewards[evader.agent_id] = -rewards[agent.agent_id]
            # rewards for the evader
            else:
                if self.current_step >= self.max_steps:
                    rewards[agent.agent_id] = self.terminal_reward
                    for pursuer in self.get_pursuer_agents():
                        rewards[pursuer.agent_id] = -rewards[agent.agent_id]
                    terminateds['__all__'] = True
                else:
                    for pursuer in self.get_pursuer_agents():
                        if pursuer.crashed:
                            rewards[agent.agent_id] = self.terminal_reward
                            rewards[pursuer.agent_id] = - \
                                rewards[agent.agent_id]
                            terminateds['__all__'] = True
                            break
                        else:
                            rewards[agent.agent_id] = self.compute_evader_reward(
                                pursuer=pursuer, evader=agent)
                            rewards[pursuer.agent_id] = - \
                                rewards[agent.agent_id]
                        old_distance: float = evader.old_distance_from_evader
                        distance_from_evader: float = evader.state_vector.distance_3D(
                            pursuer.state_vector)
                        if old_distance < distance_from_evader:
                            evader.old_distance_from_evader = distance_from_evader

        self.current_agent = next(self.agent_cycle)
        next_observations: Dict[str, np.ndarray] = self.observe(
            agent=self.get_specific_agent(self.current_agent))

        self.all_done_step += 1
        # this is a simple step counter to make sure all agents have taken a step
        if self.all_done_step >= len(self.agents):
            self.all_done_step = 0
            self.current_step += 1

        return next_observations, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.current_agent = np.random.choice(self.possible_agents)
        agent = self.get_specific_agent(self.current_agent)
        observations = self.observe(agent=agent)
        infos = {}

        return observations, infos
