import numpy as np
import gymnasium
import yaml
import numpy as np
import random

from typing import Dict, List, Optional, Text, Tuple, TypeVar
from abc import ABC, abstractmethod
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from dataclasses import dataclass, field
from aircraftsim import AircraftIC, SimInterface
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.agent import Agent, Pursuer, Evader
from jarvis.utils.vector import StateVector
from jarvis.envs.tokens import ControlIndex, ObservationIndex
from jarvis.utils.utils import normalize_obs


@dataclass
class EnvConfig:
    x_bounds: List[int] = field(default_factory=lambda: [0, 100])
    y_bounds: List[int] = field(default_factory=lambda: [0, 100])
    z_bounds: List[int] = field(default_factory=lambda: [0, 100])
    num_evaders: int = 5
    num_pursuers: int = 3
    use_pursuer_heuristics: bool = False
    dt: float = 0.1
    ai_pursuers: bool = True
    bubble_radius: int = 5
    capture_radius: float = 1.0
    min_spawn_distance: int = 10
    max_spawn_distance: int = 50
    sim_frequency: int = 100
    low_rel_pos: float = 0.0
    high_rel_pos: float = 500.0
    low_rel_vel: float = 0.0
    high_rel_vel: float = 100.0
    low_rel_att: float = 0.0
    high_rel_att: float = 2 * np.pi
    sim_end_time: float = 20
    # multiply by 1/freq * num_env_steps to get total sim time
    num_env_steps: int = int(sim_end_time*sim_frequency)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'EnvConfig':
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    # @ classmethod
    # def compute_num_steps() -> int:
    #     return int(sim_end_time*sim_frequency)


@ dataclass
class AircraftConfig:
    control_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    state_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AbstractBattleEnv(gymnasium.Env):
    def __init__(self,
                 battlespace: BattleSpace = None,
                 agents: List[Agent] = [],
                 all_agents: List[Agent] = [],
                 upload_norm_obs: bool = False,
                 vec_env: VecNormalize = None,
                 use_discrete_actions: bool = True,
                 config_file_dir: str = 'config.yaml',
                 aircraft_config_dir: str = 'aircraft_config.yaml',
                 pursuer_config_dir: str = '') -> None:
        super().__init__()
        self.battlespace: BattleSpace = battlespace
        # self.battlespace.all_agents: List[Agent] = []
        # self.agents: List[Agent] = agents  # this is the controlled agents
        # self.all_agents: List[Agent] = all_agents  # this is all agents
        self.upload_norm_obs = upload_norm_obs
        self.vec_env: VecNormalize = vec_env
        self.use_discrete_actions: bool = use_discrete_actions
        self.config_file_dir: str = config_file_dir
        self.config = self.default_config()
        self.current_step: int = 0
        self.control_limits, self.state_limits = self.load_limit_config(
            aircraft_config_dir)
        self.pursuer_config_dir: str = pursuer_config_dir
        self.pursuer_control_limits, self.pursuer_state_limits = self.load_limit_config(
            self.pursuer_config_dir)
        self.roll_commands = []
        self.altitude_commands = []
        self.velocity_commands = []

    # @ property
    # def vehicle(self) -> Agent:
    #     """First (default) controlled vehicle."""
    #     return self.agents[0] \
    #         if self.agents else None

    @property
    def all_agents(self) -> List[Agent]:
        return self.battlespace.all_agents

    @property
    def agents(self) -> List[Agent]:
        for agent in self.battlespace.all_agents:
            if agent.is_controlled:
                return agent

    def insert_agent(self, agent: Agent) -> None:
        if self.battlespace.all_agents is None:
            self.battlespace.all_agents: List[Agent] = []

        self.battlespace.all_agents.append(agent)

    def default_config(self) -> EnvConfig:
        # Path to the YAML configuration file
        config_file = 'config/' + self.config_file_dir
        try:
            # Read the YAML configuration file
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{config_file}' not found.")
        except yaml.YAMLError:
            raise ValueError(
                f"Configuration file '{config_file}' is not a valid YAML file.")

        config = EnvConfig.from_yaml(config_file)
        return config

    def load_limit_config(self, file_path: str) -> Tuple[Dict[str, Dict[str, float]],
                                                         Dict[str, Dict[str, float]]]:
        file_path = 'config/' + file_path
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{file_path}' not found.")
        except yaml.YAMLError:
            raise ValueError(
                f"Configuration file '{file_path}' is not a valid YAML file.")

        # Transform the YAML structure to the required dictionaries
        control_limits_dict = {}
        state_limits_dict = {}

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

    def simulate(self, action_dict: np.ndarray, use_multi: bool = False) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step()

    def get_agent_action_space(self, agent_id: int) -> spaces.Box:
        """
        Assumes that the action space for all agents are the same
        """
        # check if self.battlespace is not None
        if self.battlespace is None:
            raise ValueError("Battlespace is None")

        # check if agents is not None
        if self.battlespace.all_agents is None:
            raise ValueError("Agents is None")

        high_action = []
        low_action = []
        agent: Agent = self.battlespace.all_agents[agent_id]
        if agent.is_pursuer:
            action_config = self.pursuer_control_limits
            high_action, low_action = self.map_config(action_config)
        else:
            action_config = self.control_limits
            high_action, low_action = self.map_config(action_config)

        return spaces.Box(low=np.array(low_action),
                          high=np.array(high_action),
                          dtype=np.float32)

    def map_config(self, action_config: Dict) -> Tuple[List, List]:
        high = []
        low = []
        for k, v in action_config.items():
            for inner_k, inner_v in v.items():
                if 'max' in inner_k:
                    high.append(inner_v)
                elif 'min' in inner_k:
                    low.append(inner_v)

        return high, low

    def init_action_spaces(self) -> spaces.MultiDiscrete:
        """
        Assumes that you have the same action space for all agents
        If user instantiates the environment with use_discrete_actions=True
        then the action space will be a MultiDiscrete space

        For the JSBSIM application we will consider the following actions:
        - Roll command (rad)
        - Altitude command (m)
        - Velocity command (m/s)

        """
        if self.battlespace is None:
            raise ValueError("Battlespace is None")

        # check if agents is not None
        if self.battlespace.all_agents is None:
            raise ValueError("Agents is None")

        roll_idx = ControlIndex.ROLL.value
        alt_idx = ControlIndex.ALTITUDE.value
        vel_idx = ControlIndex.VELOCITY.value

        for agent in self.battlespace.all_agents:
            agent: Agent = agent
            if agent.is_controlled and self.use_discrete_actions:
                action_space = self.get_agent_action_space(
                    agent_id=agent.id)
                if self.use_discrete_actions:
                    # Discrete actions
                    self.roll_commands = np.arange(
                        action_space.low[roll_idx], action_space.high[roll_idx],
                        np.deg2rad(5))
                    self.altitude_commands = np.arange(
                        action_space.low[alt_idx], action_space.high[alt_idx], 1)
                    self.velocity_commands = np.arange(
                        action_space.low[vel_idx], action_space.high[vel_idx], 1)

                    action_space = spaces.MultiDiscrete(
                        [len(self.roll_commands),
                         len(self.altitude_commands),
                         len(self.velocity_commands)])
                    return action_space

    def discrete_to_continuous(self, action: np.ndarray) -> np.ndarray:
        """
        Convert discrete action to continuous action
        Roll command (rad)
        Altitude command (m)
        Velocity command (m/s)
        """
        roll_idx: int = ControlIndex.ROLL.value
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value

        roll_cmd: float = self.roll_commands[action[roll_idx]]
        alt_cmd: float = self.altitude_commands[action[alt_idx]]
        vel_cmd: float = self.velocity_commands[action[vel_idx]]

        return np.array([roll_cmd, alt_cmd, vel_cmd])

    @ abstractmethod
    def __init_observation_spaces(self) -> spaces.Dict:
        """
        This method needs to be implemented by the child class
        """

    @ abstractmethod
    def get_current_observation(self, agent_id: int) -> np.ndarray:
        """
        Get the current observation.
        """

    @ abstractmethod
    def __init__agents(self) -> None:
        """ This method needs to be implemented by the child class """

    @ abstractmethod
    def __init_battlespace(self) -> None:
        """
        This method needs to be implemented by the child class
        """


class DynamicThreatAvoidance(AbstractBattleEnv):
    def __init__(self,
                 battlespace: BattleSpace = None,
                 agents: List[Agent] = None,
                 upload_norm_obs: bool = False,
                 vec_env: VecNormalize = None,
                 use_discrete_actions: bool = True,
                 config_file_dir: str = 'config.yaml',
                 aircraft_config_dir: str = 'aircraft_config.yaml',
                 pursuer_config_dir: str = 'pursuer_config.yaml') -> None:
        super().__init__(battlespace=battlespace,
                         upload_norm_obs=upload_norm_obs,
                         vec_env=vec_env,
                         use_discrete_actions=use_discrete_actions,
                         config_file_dir=config_file_dir,
                         aircraft_config_dir=aircraft_config_dir,
                         pursuer_config_dir=pursuer_config_dir)
        self.__init_battlespace()
        self.__init__agents()
        self.__init_fdm()
        self.observation_space: spaces.Box = self.__init_observation_spaces()
        self.action_space: spaces.Dict = self.__init_action_spaces()
        self.old_distance_from_pursuer: float = 0.0
        self.terminal_reward: float = 1000.0

    def __getstate__(self):
        # Copy the object's state and remove unpicklable parts
        state = self.__dict__.copy()
        if 'battlespace' in state:
            del state['battlespace']
        if 'jsbsim_interface' in state:
            del state['jsbsim_interface']  # Example if JSBSim can't be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize battlespace first
        self.__init_battlespace()
        # Reinitialize agents before doing anything that requires them
        self.__init__agents()

    def __init_battlespace(self) -> None:
        """
        Creates the battlespace for the environment
        """
        # if self.battlespace:
        #     self.close_jsbsim()

        self.battlespace = BattleSpace(
            x_bounds=self.config.x_bounds,
            y_bounds=self.config.y_bounds,
            z_bounds=self.config.z_bounds,
        )

    def __init__agents(self) -> None:
        """
        Randomly spawns agents in the environment
        """

        num_pursuers = self.config.num_pursuers
        num_evaders = self.config.num_evaders
        agent_id = 0
        for i in range(num_evaders):
            rand_x = np.random.uniform(-10, 10)
            rand_y = np.random.uniform(-10, 10)
            rand_z = np.random.uniform(self.state_limits['z']['min'],
                                       self.state_limits['z']['max'])
            rand_phi = np.random.uniform(self.state_limits['phi']['min'],
                                         self.state_limits['phi']['max'])
            rand_theta = np.random.uniform(self.state_limits['theta']['min'],
                                           self.state_limits['theta']['max'])
            rand_psi = np.random.uniform(0, 2 * np.pi)
            rand_velocity = np.random.uniform(
                self.state_limits['v']['max'] - 5, self.state_limits['v']['max'])
            state_vector = StateVector(
                x=rand_x, y=rand_y, z=rand_z, roll_rad=rand_phi,
                pitch_rad=rand_theta, yaw_rad=rand_psi, speed=rand_velocity)
            print("aircraft speed", rand_velocity)
            init_conditions = AircraftIC(state_vector.x,
                                         state_vector.y,
                                         state_vector.z,
                                         state_vector.roll_rad,
                                         state_vector.pitch_rad,
                                         state_vector.yaw_rad,
                                         state_vector.speed)
            sim_interface = SimInterface(aircraft_name='x8',
                                         sim_freq=self.config.sim_frequency,
                                         init_cond=init_conditions)
            radius_bubble = self.config.bubble_radius
            evader: Evader = Evader(
                battle_space=self.battlespace,
                state_vector=state_vector,
                sim_interface=sim_interface,
                radius_bubble=radius_bubble,
                id=agent_id)
            # self.all_agents.append(evader)
            self.insert_agent(evader)
            agent_id += 1

        evader_position: StateVector = self.all_agents[0].state_vector

        for i in range(num_pursuers):
            rand_distance = np.random.uniform(
                self.config.min_spawn_distance, self.config.max_spawn_distance)
            rand_angle = np.random.uniform(0, 2 * np.pi)
            rand_x = evader_position.x + rand_distance * np.cos(rand_angle)
            rand_y = evader_position.y + rand_distance * np.sin(rand_angle)
            rand_z = evader_position.z + np.random.uniform(-10, 10)
            rand_phi = np.random.uniform(self.pursuer_state_limits['phi']['min'],
                                         self.pursuer_state_limits['phi']['max'])
            rand_theta = np.random.uniform(self.pursuer_state_limits['theta']['min'],
                                           self.pursuer_state_limits['theta']['max'])
            # rand_psi = np.random.uniform(0, 2 * np.pi)
            rand_psi = evader_position.yaw_rad + \
                np.random.uniform(-np.pi/2, np.pi/2)
            rand_velocity = np.random.uniform(
                self.pursuer_state_limits['v']['max'] - 5, self.pursuer_state_limits['v']['max'])
            state_vector = StateVector(
                x=rand_x, y=rand_y, z=rand_z, roll_rad=rand_phi,
                pitch_rad=rand_theta, yaw_rad=rand_psi, speed=rand_velocity)

            init_conditions = AircraftIC(state_vector.x,
                                         state_vector.y,
                                         state_vector.z,
                                         state_vector.roll_rad,
                                         state_vector.pitch_rad,
                                         state_vector.yaw_rad,
                                         state_vector.speed)
            sim_interface = SimInterface(aircraft_name='x8',
                                         sim_freq=self.config.sim_frequency,
                                         init_cond=init_conditions)
            pursuer: Pursuer = Pursuer(
                battle_space=self.battlespace,
                state_vector=state_vector,
                sim_interface=sim_interface,
                radius_bubble=self.config.bubble_radius,
                pursuer_state_limits=self.pursuer_state_limits,
                pursuer_control_limits=self.pursuer_control_limits,
                id=agent_id,
                capture_distance=self.config.capture_radius)
            # self.all_agents.append(pursuer)
            self.insert_agent(pursuer)
            agent_id += 1

        # self.agents = [
        #     agent for agent in self.all_agents if not agent.is_pursuer]
        # print("Number of agents", len(self.all_agents))

    def __init_fdm(self) -> None:
        if self.all_agents is None:
            raise ValueError("all_agents is not initialized!")
        for agent in self.battlespace.all_agents:
            agent.sim_interface.sim.init_fdm()

    def __init_observation_spaces(self) -> spaces.Dict:
        """
        Returns a continuous observation space for the agent in the environment
        """
        # obs_spaces = {}
        controlled_agent: Agent = self.agents
        return self.get_agent_observation_space(controlled_agent.id)

    def get_agent_observation_space(self, agent_id: int) -> spaces.Box:
        high_obs: List[float] = []
        low_obs: List[float] = []
        agent: Agent = self.battlespace.all_agents[agent_id]
        if agent.is_pursuer:
            obs_config = self.pursuer_state_limits
            high_obs, low_obs = self.map_config(obs_config)

            n_evaders = len(self.agents)
            # we need to store the relative position, heading, and velocity of evader
            # adding this into the observation space
            for i in range(n_evaders):
                low_rel_pos = self.config.low_rel_pos
                high_rel_pos = self.config.high_rel_pos
                low_rel_vel = self.config.low_rel_vel
                high_rel_vel = self.config.high_rel_vel
                low_rel_heading = self.config.low_rel_att
                high_rel_heading = self.config.high_rel_att
                low = [low_rel_pos, low_rel_vel, low_rel_heading]
                high = [high_rel_pos, high_rel_vel, high_rel_heading]
                low_obs.extend(low)
                high_obs.extend(high)
        else:
            obs_config = self.state_limits
            high_obs, low_obs = self.map_config(obs_config)

            n_pursuers = self.config.num_pursuers
            for i in range(n_pursuers):
                low_rel_pos = self.config.low_rel_pos
                high_rel_pos = self.config.high_rel_pos
                low_rel_vel = self.config.low_rel_vel
                high_rel_vel = self.config.high_rel_vel
                low_rel_heading = self.config.low_rel_att
                high_rel_heading = self.config.high_rel_att
                low = [low_rel_pos, low_rel_vel, low_rel_heading]
                high = [high_rel_pos, high_rel_vel, high_rel_heading]
                low_obs.extend(low)
                high_obs.extend(high)

        return spaces.Box(low=np.array(low_obs),
                          high=np.array(high_obs),
                          dtype=np.float32)

    def __init_action_spaces(self) -> spaces.Dict:
        return super().init_action_spaces()

    def get_current_observation(self, agent_id: int) -> np.ndarray:
        """
        Get the current observation for the agent
        For this application we will return the following:
        - x, y, z position
        - roll, pitch, yaw
        - speed
        - relative distance, heading, and velocity of evader  wrt to each pursuer
        """
        agent: Agent = self.battlespace.all_agents[agent_id]
        observation: np.array = agent.get_observation()
        # TODO: do this for pursuer as well
        agent: Evader = agent
        if agent.is_pursuer:
            raise NotImplementedError(
                "Pursuer observation not implemented yet")

        for other_agent in self.battlespace.all_agents:
            if other_agent.id == agent_id or not other_agent.is_pursuer:
                continue
            # pursuer: Pursuer = other_agent
            rel_distance: float = other_agent.distance_to(agent)
            rel_velocity: float = agent.state_vector.speed - other_agent.state_vector.speed
            rel_heading: float = other_agent.heading_difference(agent)

            # clip distance
            rel_distance: float = np.clip(rel_distance, self.config.low_rel_pos,
                                          self.config.high_rel_pos)
            rel_velocity: float = np.clip(rel_velocity, self.config.low_rel_vel,
                                          self.config.high_rel_vel)
            rel_heading: float = np.clip(rel_heading, self.config.low_rel_att,
                                         self.config.high_rel_att)

            observation = np.append(
                observation, [rel_distance, rel_velocity, rel_heading])

        # clip airspeed
        observation[ObservationIndex.AIRSPEED.value] = np.clip(
            observation[ObservationIndex.AIRSPEED.value],
            self.state_limits['v']['min'],
            self.state_limits['v']['max'])

        # assert observation.shape[0] == self.observation_space[agent_id].shape[0]
        if observation.shape[0] != self.observation_space.shape[0]:
            raise ValueError("Observation shape is not correct", observation.shape,
                             self.observation_space.shape[0])

        # check if observation is normalized
        if self.upload_norm_obs:
            observation = normalize_obs(observation, self.vec_env)
            return observation.astype(np.float32)

        return observation.astype(np.float32)

    def get_reward(self, obs: np.ndarray) -> float:
        """
        For now reward the agent for maximizing distance
        between itself and the pursuers
        """
        time_reward: float = 0.1
        controlled: Evader = self.agents
        state_obs: int = len(controlled.get_observation())
        relative_information: int = obs[state_obs:]
        rel_distances: np.array = relative_information[::3]
        # rel_velocities: np.array = relative_information[1::3]
        # rel_headings: np.array = relative_information[2::3]

        closest_distance: float = np.min(rel_distances)
        # closest_pursuer: int = np.argmin(rel_distances)

        distance_reward: float = closest_distance - self.old_distance_from_pursuer
        reward: float = distance_reward + time_reward

        return reward

    def get_info(self) -> Dict:
        info_dict: Dict = {}
        for agent in self.all_agents:
            info_dict[agent.id] = agent.get_observation()
        return info_dict

    def step(self, action: np.ndarray,
             norm_action: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        """
        truncated: bool = False
        terminated: bool = False
        info = self.get_info()

        if self.use_discrete_actions:
            action: np.ndarray = self.discrete_to_continuous(action)
        self.simulate(action, use_multi=False)
        controlled_agent: Evader = self.agents
        obs: np.array = self.get_current_observation(self.agents.id)
        reward: float = self.get_reward(obs)

        for agent in self.all_agents:
            agent: Agent
            if agent.crashed:
                print("You died")
                reward -= self.terminal_reward
                terminated = True
                truncated = True

        self.current_step += 1
        if self.current_step >= self.config.num_env_steps:
            print("You win")
            reward += self.terminal_reward
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = 'human') -> None:
        pass

    def reinit_jsbsim(self) -> None:
        """
        To make this thread safe we will only update the
        simulation environment and not close it
        TODO: need to refactor a lot of code duplication
        """

        for agent in self.all_agents:
            if agent.is_pursuer:
                continue
            agent: Evader
            rand_x = np.random.uniform(-10, 10)
            rand_y = np.random.uniform(-10, 10)
            rand_z = np.random.uniform(self.state_limits['z']['min'],
                                       self.state_limits['z']['max'])
            rand_phi = np.random.uniform(self.state_limits['phi']['min'],
                                         self.state_limits['phi']['max'])
            rand_theta = np.random.uniform(self.state_limits['theta']['min'],
                                           self.state_limits['theta']['max'])
            rand_psi = np.random.uniform(0, 2 * np.pi)
            rand_velocity = np.random.uniform(
                self.state_limits['v']['max'] - 5, self.state_limits['v']['max'])
            state_vector = StateVector(
                x=rand_x, y=rand_y, z=rand_z, roll_rad=rand_phi,
                pitch_rad=rand_theta, yaw_rad=rand_psi, speed=rand_velocity)
            agent.state_vector: StateVector = state_vector
            init_conditions = AircraftIC(state_vector.x,
                                         state_vector.y,
                                         state_vector.z,
                                         state_vector.roll_rad,
                                         state_vector.pitch_rad,
                                         state_vector.yaw_rad,
                                         state_vector.speed)
            agent.sim_interface.sim.reinitialise(init_conditions)

        evader_position = self.all_agents[0].state_vector

        for agent in self.all_agents:
            agent: Agent
            if agent.is_pursuer:
                rand_distance = np.random.uniform(
                    self.config.min_spawn_distance, self.config.max_spawn_distance)
                rand_angle = np.random.uniform(0, 2 * np.pi)
                rand_x = evader_position.x + rand_distance * np.cos(rand_angle)
                rand_y = evader_position.y + rand_distance * np.sin(rand_angle)
                rand_z = evader_position.z + np.random.uniform(-10, 10)
                rand_phi = np.random.uniform(self.pursuer_state_limits['phi']['min'],
                                             self.pursuer_state_limits['phi']['max'])
                rand_theta = np.random.uniform(self.pursuer_state_limits['theta']['min'],
                                               self.pursuer_state_limits['theta']['max'])
                # rand_psi = np.random.uniform(0, 2 * np.pi)
                rand_psi = evader_position.yaw_rad + \
                    np.random.uniform(-np.pi/2, np.pi/2)
                rand_velocity = np.random.uniform(
                    self.pursuer_state_limits['v']['max'] - 5, self.pursuer_state_limits['v']['max'])
                state_vector = StateVector(
                    x=rand_x, y=rand_y, z=rand_z, roll_rad=rand_phi,
                    pitch_rad=rand_theta, yaw_rad=rand_psi, speed=rand_velocity)
                init_conditions = AircraftIC(state_vector.x,
                                             state_vector.y,
                                             state_vector.z,
                                             state_vector.roll_rad,
                                             state_vector.pitch_rad,
                                             state_vector.yaw_rad,
                                             state_vector.speed)
                agent.state_vector: StateVector = state_vector
                agent.sim_interface.reset_sim(init_conditions)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        This method needs to be implemented by the child class
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Clear any previous state
        self.battlespace.clear_jsbsim()
        self.battlespace.clear_agents()

        # Reinitialize agents and then the FDM
        self.__init__agents()
        self.__init_fdm()  # Ensure FDM is initialized only after agents exist

        self.old_distance_from_pursuer = 0.0
        self.current_step: int = 0
        controlled_agent: Evader = self.agents
        observation = self.get_current_observation(controlled_agent.id)
        infos = self.get_info()
        return observation, infos
