import yaml
import gymnasium as gym
import numpy as np
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


class AbstractKinematicEnv(MultiAgentEnv, ABC):
    """
    """

    def __init__(self, config: Optional[Dict[str, Any]], **kwargs) -> None:
        # Call the next __init__ in the MRO (likely gym.Env.__init__)
        super().__init__(**kwargs)
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
        # self.roll_commands: np.array = None
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
            str(agent.agent_id) for agent in self.get_controlled_agents]

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

    # def insert_agent(self, agent: SimpleAgent) -> None:
    #     if agent.is_controlled:
    #         self.agents.append(agent.agent_id)

    #     self.battlespace.all_agents.append(agent)
    #     self.agents = [str(agent.agent_id)
    #                    for agent in self.get_controlled_agents]

    def build(self) -> None:
        self.__init_battlespace()
        assert self.battlespace is not None
        self.__init__agents()
        self.agents: List[int] = [
            str(agent.agent_id) for agent in self.get_controlled_agents]

    def get_specific_agent(self, agent_id: str) -> SimpleAgent:
        """
        Returns a specific agent in the environment
        """
        agent_id = str(agent_id)
        for agent in self.battlespace.all_agents:
            if agent.agent_id == agent_id:
                return agent

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

    def spawn_agents(self,
                     num_agents: int,
                     agent_id: str,
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
                state_limits['z']['min']+20, state_limits['z']['max']-20)

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
            if not is_pursuer:
                agent: Evader = Evader(
                    battle_space=self.battlespace,
                    state_vector=state_vector,
                    simple_model=plane_model,
                    radius_bubble=radius_bubble,
                    is_controlled=is_controlled,
                    agent_id=str(agent_id))

            else:
                capture_radius: float = self.agent_config['interaction']['capture_radius']
                agent = Pursuer(
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

    def insert_agent(self, agent: SimpleAgent,
                     place_index: int = None) -> None:
        """
        Args: 
            agent (SimpleAgent): The agent to insert into the environment
            place_at_start (bool): True if we want to place the agent at the 
            start of the list, False otherwise
        Inserts an agent into the environment
        if list of all agents is not initialized
        then we initialize it
        """
        if self.battlespace.all_agents is None:
            self.battlespace.all_agents: List[SimpleAgent] = []

        if place_index:
            self.battlespace.all_agents.insert(place_index, agent)
        else:
            self.battlespace.all_agents.append(agent)

    def remove_agent(self, agent_id: str) -> None:
        """
        Args:
            agent_id (str): The id of the agent to remove
        Removes an agent from the environment
        """
        agent_id = str(agent_id)
        for i, agent in enumerate(self.battlespace.all_agents):
            if agent.agent_id == agent_id:
                self.battlespace.all_agents.pop(i)

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
            - 7:vx: The x velocity of the agent
            - 8:vy: The y velocity of the agent
            - 9:vz: The z velocity of the agent

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
        if self.pitch_commands is None or self.yaw_commands is None or self.airspeed_commands is None:
            raise ValueError(
                "The roll, pitch, yaw, and airspeed commands are not initialized")

        pitch_idx: int = action[0]
        yaw_idx: int = action[1]
        vel_idx: int = action[2]

        pitch_cmd: float = self.pitch_commands[pitch_idx]
        yaw_cmd: float = self.yaw_commands[yaw_idx]
        vel_cmd: float = self.airspeed_commands[vel_idx]

        return np.array([pitch_cmd, yaw_cmd, vel_cmd], dtype=np.float32)

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
        self.pitch_commands: np.array = np.arange(
            continous_action_space.low[pitch_idx], continous_action_space.high[pitch_idx],
            np.deg2rad(1))
        self.yaw_commands: np.array = np.arange(
            continous_action_space.low[yaw_idx], continous_action_space.high[yaw_idx],
            np.deg2rad(1))
        self.airspeed_commands: np.array = np.arange(
            continous_action_space.low[vel_idx], continous_action_space.high[vel_idx],
            1)

        action_space = gym.spaces.MultiDiscrete([len(self.pitch_commands),
                                                len(self.yaw_commands),
                                                len(self.airspeed_commands)])

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
                            projection_time: float = 1.0) -> np.ndarray:
        """
        Args:
            agent (SimpleAgent): The agent to mask the pitch commands for
            action_mask (np.ndaaction[]rray): The action mask to update
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

        # get current pitch angle
        pitch_rad: float = agent.state_vector.pitch_rad
        # pitch_idx: int = np.argmin(np.abs(self.pitch_commands - pitch_rad))
        # mask values outside 10 degrees of the current pitch angle
        pitch_mask = np.ones_like(self.pitch_commands, dtype=np.int8)
        pitch_mask = np.where(
            np.abs(self.pitch_commands - pitch_rad) > np.deg2rad(10), 0, pitch_mask)

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

        # for i, yaw_cmd in enumerate(self.yaw_commands):
        #     u: np.ndarray = np.array([0,
        #                               0,
        #                               yaw_cmd,
        #                               current_speed])
        #     # next_state: np.array = agent.simple_model.rk45(
        #     #     x=current_state, u=u, dt=projection_time)
        #     # x_proj: float = next_state[KinematicIndex.X.value]
        #     # y_proj: float = next_state[KinematicIndex.Y.value]
        #     x_proj: float = agent.state_vector.x + \
        #         (current_speed * np.cos(yaw_cmd) * projection_time)
        #     y_proj: float = agent.state_vector.y + \
        #         (current_speed * np.sin(yaw_cmd) * projection_time)
        #     if x_proj > x_bounds['max'] or x_proj < x_bounds['min']:
        #         yaw_mask[i] = 0
        #     if y_proj > y_bounds['max'] or y_proj < y_bounds['min']:
        #         yaw_mask[i] = 0

        return yaw_mask

    def unwrap_action_mask(self, action_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits the action mask into roll, pitch, yaw, and velocity
        masks
        """
        # unwrapped_mask: Dict[str, np.array] = {
        #     'roll': action_mask[:len(self.roll_commands)],
        #     'pitch': action_mask[len(self.roll_commands):len(self.roll_commands) + len(self.pitch_commands)],
        #     'yaw': action_mask[len(self.roll_commands) + len(self.pitch_commands):len(self.roll_commands) + len(self.pitch_commands) + len(self.yaw_commands)],
        #     'vel': action_mask[len(self.roll_commands) + len(self.pitch_commands) + len(self.yaw_commands):]
        # }

        unwrapped_mask: Dict[str, np.array] = {
            'pitch': action_mask[:len(self.pitch_commands)],
            'yaw': action_mask[len(self.pitch_commands):len(self.pitch_commands) + len(self.yaw_commands)],
            'vel': action_mask[len(self.pitch_commands) + len(self.yaw_commands):]
        }

        return unwrapped_mask

    def wrap_action_mask(self, unwrapped_mask: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Wraps the action mask into a single array
        """
        # action_mask = np.concatenate([unwrapped_mask['roll'],
        #                               unwrapped_mask['pitch'],
        #                               unwrapped_mask['yaw'],
        #                               unwrapped_mask['vel']])

        action_mask = np.concatenate([unwrapped_mask['pitch'],
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
        # roll_mask: np.array = np.ones_like(self.roll_commands, dtype=np.int8)
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

        full_mask = np.concatenate([pitch_mask, yaw_mask, vel_mask])

        if action_space_sum is not None:
            assert full_mask.size == action_space_sum
        else:
            assert full_mask.size == self.action_space.nvec.sum()

        return full_mask

    def simulate(self, action_dict: np.ndarray, use_multi: bool = False) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step()

    def simulate_single(self, agent: SimpleAgent, action: np.ndarray) -> None:
        self.battlespace.act(action, use_multi=True)
        self.battlespace.step_single_agent(agent=agent)

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

        # make sure obs is np.float32t("e")
        obs = np.array(obs, dtype=np.float32)
        action_mask: np.ndarray = self.get_action_mask(agent=agent,
                                                       action_space_sum=total_actions)
        return {'observations': obs, 'action_mask': action_mask}


class PursuerEvaderEnv(AbstractKinematicEnv):
    """
    Evader where must avoid being captured by the pursuers must 
    get to the target location 
    """

    def __init__(self, config: Optional[Dict]) -> None:
        super(PursuerEvaderEnv, self).__init__(config=config)

        self.interaction_config: Dict[str,
                                      Any] = self.agent_config['interaction']
        self.relative_state_observations: Dict[str,
                                               Any] = self.agent_config['relative_state_observations']
        self.action_spaces = self.init_action_space()
        self.observation_spaces = self.init_observation_space()
        self.current_step: int = 0
        self.possible_agents: List[int] = self.agents
        self.agent_cycle = itertools.cycle(self.possible_agents)
        self.current_agent: str = next(self.agent_cycle)
        self.terminal_reward: float = 1000.0
        self.all_done_step: int = 0
        self.use_pronav: bool = self.agent_config['use_pronav']

        # self.__init_battlespace()
        assert self.battlespace is not None
        self.__init__agents()
        self.agents: List[int] = [
            str(agent.agent_id) for agent in self.get_controlled_agents]

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

    def observe(self, agent: SimpleAgent,
                num_actions: int = None) -> Dict[str, np.ndarray]:
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
        observation: Dict[str, np.ndarray] = super().observe(
            agent, num_actions)
        obs: np.ndarray = observation['observations']
        action_mask: np.ndarray = observation['action_mask']

        overall_relative_pos: List[float] = []
        for other_agent in self.get_controlled_agents:
            if agent.agent_id == other_agent.agent_id:
                continue

            # we don't want to include the other pursuer in the observation
            if agent.is_pursuer == other_agent.is_pursuer:
                continue

            #TODO: clean this up? abstract to a method?
            if agent.is_pursuer and not self.use_pronav:
                unpacked_actions: Dict[str, np.ndarray] = self.unwrap_action_mask(
                    action_mask)
                pronav: ProNavV2 = ProNavV2()
                current_pos = agent.state_vector.array[0:3]
                evader: Evader = self.get_evader_agents()[0]
                target_pos = evader.state_vector.array[0:3]
                relative_pos = target_pos - current_pos
                relative_vel = evader.state_vector.speed - \
                    agent.state_vector.speed
                action = pronav.predict(
                    current_pos=current_pos,
                    relative_pos=relative_pos,
                    current_heading=agent.state_vector.yaw_rad,
                    current_speed=agent.state_vector.speed,
                    relative_vel=relative_vel
                )
                yaw_idx:int = 1
                yaw_cmd:float = action[yaw_idx]
                #yaw_cmd_index:int = np.abs(self.yaw_commands - yaw_cmd).argmin()
                yaw_cmd_index:int = np.abs(self.yaw_commands - yaw_cmd).argmin()
                # TODO: Make the FOV a parameter use can update
                fov_half:int = 10
                indices = np.arange(yaw_cmd_index - fov_half, yaw_cmd_index + fov_half + 1) \
                    % len(self.yaw_commands)
                #clip the indices to min and max values
                yaw_mask:np.array = np.zeros_like(self.yaw_commands)
                yaw_mask[indices] = 1
                unpacked_actions['yaw'] = yaw_mask
                # sanity check
                assert(yaw_mask[yaw_cmd_index] == 1)
                # we're going to update the pitch masks with the consideration
                # of where the evader is, we want to null out pitch commands outside
                # the FOV in the pitch 
                pitch_desired:float = self.adjust_pitch(
                    selected_agent=agent,
                    evader=evader
                )
                fov_half:int = 5
                pitch_mask: np.array = unpacked_actions['pitch']
                pitch_cmd_index = np.abs(self.pitch_commands - pitch_desired).argmin()
                pitch_indices = np.arange(pitch_cmd_index - fov_half, pitch_cmd_index + fov_half + 1) \
                    % len(self.pitch_commands)
                # get all indicies that are 0 currently
                new_mask = np.zeros_like(pitch_mask)
                indices = np.where(pitch_mask == 0)[0]
                updated_indices = np.setdiff1d(pitch_indices, indices)
                new_mask[updated_indices] = 1
                # everything else is 0
                unpacked_actions['pitch'] = new_mask
                action_mask = self.wrap_action_mask(unpacked_actions)

            relative_pos: np.ndarray = agent.state_vector.array - \
                other_agent.state_vector.array
            relative_velocity = agent.state_vector.speed - \
                other_agent.state_vector.speed
            relative_heading = agent.state_vector.yaw_rad - \
                other_agent.state_vector.yaw_rad
            # wrap heading between -pi and pi
            relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi

            relative_pos = relative_pos[:3]

            relative_info = [relative_pos[0], relative_pos[1], relative_pos[2],
                             relative_velocity, relative_heading]

            overall_relative_pos.extend(relative_info)

        obs = np.concatenate([obs, overall_relative_pos]).astype(np.float32)
        observation['observations'] = obs
        observation['action_mask'] = action_mask
        # check to make sure
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

        return delta_distance + (0.5 *dot_product)

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
        return - delta_distance - (0.5 *dot_product)

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

    def adjust_pitch(self, selected_agent: Pursuer,
                     evader: SimpleAgent,
                     target_instead: bool = False,
                     target_statevector: StateVector = None) -> float:
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
        # action[pitch_idx] = -pitch_cmd

        return -pitch_cmd

    def step(self, action_dict: Dict[str, Any],
             specific_agent_id: int = None,
             use_pronav: bool = False) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        """
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        rewards = {agent: 0.0 for agent in self.agents}
        infos = {}
        observations = {}

        if specific_agent_id is None:
            # action: np.array = self.discrete_to_continuous_action(
            #     action_dict[selected_agent.agent_id]['action'])
            # command_action: Dict[str, np.array] = {
            #     selected_agent.agent_id: action
            selected_agent: SimpleAgent = self.get_specific_agent(
                self.current_agent)
            selected_agent_id = selected_agent.agent_id
        else:
            selected_agent_id = specific_agent_id
            selected_agent: SimpleAgent = self.get_specific_agent(
                specific_agent_id)

        action: np.array = self.discrete_to_continuous_action(
            action_dict[str(selected_agent_id)]['action'])

        # if selected_agent.is_pursuer:
        #     action = self.adjust_pitch(
        #         selected_agent, self.get_evader_agents()[0], action)

        if selected_agent.is_pursuer:
            if self.use_pronav:
                pronav: ProNavV2 = ProNavV2()
                current_pos = selected_agent.state_vector.array[0:3]
                evader: Evader = self.get_evader_agents()[0]
                target_pos = evader.state_vector.array[0:3]
                relative_pos = target_pos - current_pos
                relative_vel = evader.state_vector.speed - \
                    selected_agent.state_vector.speed
                vel_cmd = action[-1]
                action = pronav.predict(
                    current_pos=current_pos,
                    relative_pos=relative_pos,
                    current_heading=selected_agent.state_vector.yaw_rad,
                    current_speed=selected_agent.state_vector.speed,
                    relative_vel=relative_vel,
                    consider_yaw=False,
                    max_vel=vel_cmd
                )

        command_action: Dict[str, np.array] = {selected_agent.agent_id: action}
        self.simulate(command_action, use_multi=True)
        # self.simulate_single(agent=selected_agent, action=action)

        num_actions = self.action_spaces[selected_agent.agent_id]["action"].nvec.sum(
        )
        observations = self.observe(
            agent=selected_agent, num_actions=num_actions)
        evaders: List[SimpleAgent] = self.get_evader_agents()
        evader = evaders[0]

        for agent in self.get_controlled_agents:
            # rewards for the pursuers
            if agent.is_pursuer:
                if self.is_caught(pursuer=agent, evader=evader) or evader.crashed:
                    print("Evader,   Caught")
                    terminateds['__all__'] = True
                    rewards[evader.agent_id] = -self.terminal_reward
                    for pursuer in self.get_pursuer_agents():
                        rewards[pursuer.agent_id] = self.terminal_reward
                    # game is over
                    break
                else:
                    # compute the intermediate reward
                    rewards[agent.agent_id] = self.compute_pursuer_reward(
                        pursuer=agent, evader=evader)
                    rewards[evader.agent_id] = -rewards[agent.agent_id]
            # rewards for the evader
            else:
                if self.current_step >= self.max_steps:
                    terminateds['__all__'] = True
                    rewards[evader.agent_id] = self.terminal_reward
                    for pursuer in self.get_pursuer_agents():
                        rewards[pursuer.agent_id] = -rewards[evader.agent_id]
                    print("Evader Won", rewards)
                    break
                else:
                    for pursuer in self.get_pursuer_agents():
                        # get closest pursuer
                        min_distance = 1000
                        distance = pursuer.state_vector.distance_3D(
                            evader.state_vector)

                        if distance < min_distance:
                            min_distance = distance
                            evader.old_distance_from_pursuer = min_distance
                            # closest pursuer
                            rewards[evader.agent_id] = self.compute_evader_reward(
                                pursuer=pursuer, evader=evader)
                            rewards[pursuer.agent_id] = -\
                                rewards[evader.agent_id]

                        if pursuer.crashed and not evader.crashed:
                            terminateds['__all__'] = True
                            rewards[evader.agent_id] = self.terminal_reward
                            rewards[pursuer.agent_id] = - \
                                rewards[agent.agent_id]
                            print("Pursuer Crashed", pursuer.state_vector)

        self.current_agent = next(self.agent_cycle)
        # check if key exists
        if self.current_agent not in self.action_spaces:
            while self.current_agent not in self.action_spaces:
                self.current_agent = next(self.agent_cycle)

        num_actions: int = self.action_spaces[self.current_agent]["action"].nvec.sum(
        )

        next_observations: Dict[str, np.ndarray] = {}
        next_observations[self.current_agent] = self.observe(
            agent=self.get_specific_agent(self.current_agent),
            num_actions=num_actions)
        self.all_done_step += 1
        # this is a simple step counter to make sure all agents have taken a step
        if self.all_done_step >= len(self.agents):
            self.all_done_step = 0
            self.current_step += 1
        return next_observations, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        assert self.battlespace is not None
        self.__init__agents()
        self.agents: List[int] = [
            str(agent.agent_id) for agent in self.get_controlled_agents]
        self.current_agent = next(self.agent_cycle)

        agent: SimpleAgent = self.get_specific_agent(self.current_agent)
        num_actions: int = self.action_spaces[agent.agent_id]["action"].nvec.sum(
        )
        observations: Dict[str, np.ndarray] = {}
        observations[self.current_agent] = self.observe(
            agent=agent, num_actions=num_actions)
        infos = {}

        return observations, infos
