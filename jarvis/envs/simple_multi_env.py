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
from jarvis.envs.tokens import KinematicIndex
# abstract methods
from abc import ABC, abstractmethod


"""
Initialize variables
Initialize optimizer
Init criteria

"""


class AbstracKinematicEnv(gym.Env, ABC):
    """

    """

    def __init__(self,
                 config: Optional[Dict]) -> None:
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
            # control_limits = self.evader_control_limits
            rand_x = 0.0  # np.random.uniform(-10, 10)
            rand_y = 0.0  # np.random.uniform(-10, 10)
            # np.random.uniform(state_limits['z']['min'],state_limits['z']['max'])
            rand_z = 50.0

            rand_phi = np.random.uniform(state_limits['phi']['min'],
                                         state_limits['phi']['max'])
            rand_theta = np.random.uniform(state_limits['theta']['min'],
                                           state_limits['theta']['max'])
            rand_psi = 0.0  # np.random.uniform(0, 2 * np.pi)
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

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, vel_cmd])

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

    def get_action_mask(self, agent: SimpleAgent) -> np.ndarray:
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
        roll_mask: np.array = np.ones_like(self.roll_commands)
        pitch_mask: np.array = np.ones_like(self.pitch_commands)
        yaw_mask: np.array = np.ones_like(self.yaw_commands)
        vel_mask: np.array = np.ones_like(self.airspeed_commands)

        pitch_mask: np.array = self.mask_pitch_commands(agent=agent,
                                                        pitch_mask=pitch_mask,
                                                        z_bounds=z_bounds)

        yaw_mask: np.array = self.mask_psi_commands(agent=agent,
                                                    yaw_mask=yaw_mask,
                                                    x_bounds=x_bounds,
                                                    y_bounds=y_bounds)

        full_mask = np.concatenate([roll_mask, pitch_mask, yaw_mask, vel_mask])

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

    def observe(self, agent: SimpleAgent) -> Dict[str, np.ndarray]:
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

        Refer to get_observation_space for the observation space definition.
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

        obs_space: Dict[str, gym.spaces.Box] = self.get_observation_space(
            is_pursuer=agent.is_pursuer)
        low = obs_space['observations'].low
        high = obs_space['observations'].high

        if np.any(obs < low) or np.any(obs > high):
            # print the one out of bounds
            for i, (obs_val, low_val, high_val) in enumerate(zip(obs, low, high)):
                if obs_val < low_val or obs_val > high_val:
                    raise ValueError("Observation out of bounds",
                                     f"Observation {i} out of bounds: {obs_val} not in [{low_val}, {high_val}]")

        action_mask: np.ndarray = self.get_action_mask(agent=agent)
        return {'observations': obs, 'action_mask': action_mask}


class EngageEnv(AbstracKinematicEnv):
    def __init__(self, config: Optional[Dict]) -> None:
        super(EngageEnv, self).__init__(config=config)
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

        observation_space = self.get_observation_space(is_pursuer=True)
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
        """
        capture_radius: float = self.agent_interaction['capture_radius']

        distance: float = self.target.distance_3D(agent.state_vector)
        if distance <= capture_radius:
            return True

        return False

    def observe(self, agent) -> Dict[str, np.ndarray]:
        observation: Dict[str, np.ndarray] = super().observe(agent)
        obs: np.ndarray = observation['observations']

        target_obs: np.ndarray = self.target.array
        relative_pos: np.ndarray = target_obs[:3] - obs[:3]
        # clip the relative position
        low = self.relative_observations['position']['x']['low']
        high = self.relative_observations['position']['x']['high']
        relative_pos = np.clip(relative_pos, low, high)

        obs = np.concatenate([obs, relative_pos])
        observation['observations'] = obs

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
            self.ctrl_counter = 0
            self.old_action: np.ndarray = action
        else:
            self.ctrl_counter += 1
            self.simulate(self.old_action, use_multi=False)

        agent: SimpleAgent = self.get_controlled_agents[0]
        observation: Dict[str, np.ndarray] = self.observe(agent=agent)
        reward: float = self.compute_intermediate_reward(agent=agent)

        if self.current_step >= self.max_steps:
            self.terminal_reward = -self.terminal_reward
            terminated: bool = True
        elif self.is_close_to_target(agent=self.get_controlled_agents[0]):
            reward = self.terminal_reward
            terminated: bool = True

        for agent in self.get_controlled_agents:
            distance: float = self.target.distance_3D(agent.state_vector)
            info[agent.agent_id] = {
                'crashed': agent.crashed,
                'reward': reward,
                'distance_to_goal': distance
            }

        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.__init_target()


class AvoidEnv(AbstracKinematicEnv):
    def __init__agents(self):
        return super().__init__agents()
