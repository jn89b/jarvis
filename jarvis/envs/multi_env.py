"""
Create a base environment for multi-agent reinforcement learning.
- Train policy for agent to engage a specific target
- Train policy for agent to avoid n pursuers
    - pursuers can use a heuristic or a trained policy
- Train a meta policy for the pursuers to capture the evader
"""
import yaml
import gymnasium as gym
import numpy as np
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from jarvis.envs.agent import Agent, Pursuer, Evader
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector
from jarvis.envs.tokens import ControlIndex, ObservationIndex

from aircraftsim import AircraftIC, SimInterface


@dataclass
class EnvConfig:
    """
    This is a standard data class to read a yaml file and store the environment configuration.
    Used to set up the environment for training and testing.
    """
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
    sim_frequency: int = 100  # Hz

    low_rel_x: float = -750.0
    high_rel_x: float = 750.0

    low_rel_y: float = -750.0
    high_rel_y: float = 750.0

    low_rel_z: float = -50.0
    high_rel_z: float = 50.0

    low_rel_pos: float = 0.0
    high_rel_pos: float = 500.0

    low_rel_vel: float = 0.0
    high_rel_vel: float = 100.0

    low_rel_att: float = 0.0

    high_rel_att: float = 2 * np.pi
    sim_end_time: float = 50.0  # this the actual time in the simulation

    # multiply by 1/freq * num_env_steps to get total sim time
    num_env_steps: int = int(sim_end_time*sim_frequency)

    randomize_target: bool = True
    target_x: int = None
    target_y: int = None
    target_z: int = None
    target_spawn_radius_min: float = 250
    target_spawn_radius_max: float = 400

    @classmethod
    def from_yaml(cls, file_path: str) -> 'EnvConfig':
        cwd = os.getcwd()
        # log the current working directory
        full_file_path = os.path.join(cwd, file_path)
        print("Current working directory:", cwd)
        if not os.path.exists(full_file_path):
            return cls()
        else:
            with open(full_file_path, 'r') as f:
                config_data = yaml.safe_load(f)

        return cls(**config_data)


@dataclass
class AircraftConfig:
    control_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    state_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)


def load_limit_config(file_path: str) -> Tuple[Dict[str, Dict[str, float]],
                                               Dict[str, Dict[str, float]]]:
    """
    Reads the configuration file and returns the control and state limits
    for the agents in the environment.

    Args:
        file_path: The path to the configuration file.

    Returns:
        A tuple containing the control and state limits for the agents.
        control_limits_dict: A dictionary containing the control limits for each agent.
        state_limits_dict: A dictionary containing the state limits for each agent.

        Each dictionary has the following structure:
        control_limits_dict = {
            'u_phi': {'min': min_val, 'max': max_val},
            'u_theta': {'min': min_val, 'max': max_val},
            ... }
        state_limits_dict = {
            'x': {'min': min_val, 'max': max_val},
            'y': {'min': min_val, 'max': max_val},
            ... }
    """

    # file_path = 'config/' + file_path
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


class TargetEngageEnv(gym.Env):
    """
    Environment to approach a target within some radius

    Agents are aircraft that are controlled by the user/policy
    All agents are aircraft that are in the environment

    Implements multidiscrete masking
    https://discuss.ray.io/t/is-any-multi-discrete-action-example-for-ppo-or-other-algorithms/4693/6
    """

    def __init__(
        self,
        battlespace: BattleSpace = None,
        agents: List[Agent] = None,
        upload_norm_obs: bool = False,
        use_discrete_actions: bool = True,
        config_file_dir: EnvConfig = None,
        aircraft_config_dir: str = 'config/aircraft_config.yaml',
        control_limits: Dict[str, Dict[str, float]] = None,
        state_limits: Dict[str, Dict[str, float]] = None
    ):
        # self.config = EnvConfig.from_yaml(config_file_dir)
        self.config = config_file_dir
        self.aircraft_config_dir: str = aircraft_config_dir
        if control_limits is None or state_limits is None:
            self.control_limits, self.state_limits = load_limit_config(
                aircraft_config_dir)
        else:
            self.control_limits = control_limits
            self.state_limits = state_limits

        if battlespace is None or not isinstance(battlespace, BattleSpace):
            self.__init_battlespace()
        else:
            self.battlespace = battlespace

        # if agents is None:
        #     self.__init__agents()
        # else:
        #     self.get_controlled_agents = agents

        self.__init__agents()
        if self.get_controlled_agents is None:
            raise ValueError("Controlled agents not initialized!")

        self.__init_target()
        self.upload_norm_obs = upload_norm_obs
        self.use_discrete_actions = use_discrete_actions
        self.current_step: int = 0
        self.agents = [agent.id for agent in self.get_controlled_agents]

        self.roll_commands: np.array = None
        self.alt_commands: np.array = None
        self.airspeed_commands: np.array = None
        self.action_space: gym.spaces.MultiDiscrete = self.get_discrete_action_space()
        self.observation_space: gym.spaces.Box = self.get_observation_space()

        # this is used to control how often the agent can update its high level control
        self.old_action: np.array = None
        self.ctrl_time: float = 0.2  # control time in seconds
        self.ctrl_time_index = int(self.ctrl_time * self.config.sim_frequency)
        self.ctrl_counter: int = 0  # control counter

        # used for reward calculation
        self.old_distance_from_target: float = self.target.distance_3D(
            self.get_controlled_agents[0].state_vector
        )
        self.old_dot_product: float = self.target.dot_product_2D(
            self.get_controlled_agents[0].state_vector
        )
        self.terminal_reward: float = 100.0

    @property
    def get_all_agents(self) -> List[Agent]:
        return self.battlespace.all_agents

    @property
    def get_controlled_agents(self) -> List[Agent]:
        """
        Returns all the controlled agents in the environment
        """
        controlled_agents = []
        for agent in self.battlespace.all_agents:
            if agent.is_controlled:
                controlled_agents.append(agent)

        return controlled_agents

    def get_observation_space(self) -> gym.spaces.Dict:
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
            - 7:dx: The relative x position of the agent to the target (target_x - agent_x)
            - 8:dy: The relative y position of the agent to the target (target_y - agent_y)
            - 9:dz: The relative z position of the agent to the target (target_z - agent_z)
        """
        obs_config = self.state_limits
        high_obs, low_obs = self.map_config(obs_config)
        high_obs = high_obs[:7]
        low_obs = low_obs[:7]

        low_rel_x = self.config.low_rel_x
        high_rel_x = self.config.high_rel_x

        low_rel_y = self.config.low_rel_y
        high_rel_y = self.config.high_rel_y

        low_rel_z = self.config.low_rel_z
        high_rel_z = self.config.high_rel_z

        low = [low_rel_x, low_rel_y, low_rel_z]
        high = [high_rel_x, high_rel_y, high_rel_z]

        low_obs.extend(low)
        high_obs.extend(high)

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

    def map_config(self, control_config: Dict) -> Tuple[List, List]:
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

    def get_continous_action_space(self) -> gym.spaces.Box:
        """
        Initializes the action space for the environment
        """
        # agent: Agent = self.get_controlled_agents[0]
        high, low = self.map_config(self.control_limits)

        return gym.spaces.Box(low=np.array(low),
                              high=np.array(high),
                              dtype=np.float32)

    def discrete_to_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """

        Args:
            action: The discrete action to convert to continuous action,
            refer to get_discrete_action_space for the action space definition.

        Returns:
            The continuous action in the form of [roll_cmd, alt_cmd, vel_cmd].

        Convert discrete action to continuous action
        Roll command (rad)
        Altitude command (Absolute) (m)
        Velocity command (m/s)

        """

        if self.roll_commands is None or self.alt_commands is None or self.airspeed_commands is None:
            raise ValueError(
                "Commands not initialized, please initialize commands")

        roll_idx: int = ControlIndex.ROLL.value
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value

        roll_cmd: float = self.roll_commands[action[roll_idx]]
        alt_cmd: float = self.alt_commands[action[alt_idx]]
        vel_cmd: float = self.airspeed_commands[action[vel_idx]]

        return np.array([roll_cmd, alt_cmd, vel_cmd])

    def get_discrete_action_space(self) -> gym.spaces.MultiDiscrete:
        """
        Args:
            None

        Returns:
            The action space for the environment.

        Initializes the action space for the environment
        For this environment the action space will be a discrete space
        where the aircraft can send commands in the form of roll, dz, velocity

        action_space:
            [
                roll_cmd_idx:[0, 1, 2, ...n],
                alt_cmd_idx:[0, 1, 2, ...n],
                vel_cmd_idx:[0, 1, 2, ...n]
            ]

        This is mapped to the continous action space
        To get the actual commands use the discrete_to_continuous_action method
        """
        roll_idx: int = ControlIndex.ROLL.value
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value
        continous_action_space: gym.spaces.Box = self.get_continous_action_space()

        self.roll_commands: np.array = np.arange(
            continous_action_space.low[roll_idx], continous_action_space.high[roll_idx],
            np.deg2rad(5))
        self.alt_commands: np.array = np.arange(
            continous_action_space.low[alt_idx], continous_action_space.high[alt_idx],
            1)
        self.airspeed_commands: np.array = np.arange(
            continous_action_space.low[vel_idx], continous_action_space.high[vel_idx],
            1)

        action_space = gym.spaces.MultiDiscrete(
            [len(self.roll_commands), len(self.alt_commands), len(self.airspeed_commands)])

        return action_space

    def insert_agent(self, agent: Agent) -> None:
        if self.battlespace.all_agents is None:
            self.battlespace.all_agents: List[Agent] = []

        self.battlespace.all_agents.append(agent)

    @classmethod
    def load_limit_config(file_path: str) -> Tuple[Dict[str, Dict[str, float]],
                                                   Dict[str, Dict[str, float]]]:
        """
        Reads the configuration file and returns the control and state limits
        for the agents in the environment.

        Args:
            file_path: The path to the configuration file.

        Returns:
            A tuple containing the control and state limits for the agents.
            control_limits_dict: A dictionary containing the control limits for each agent.
            state_limits_dict: A dictionary containing the state limits for each agent.

            Each dictionary has the following structure:
            control_limits_dict = {
                'u_phi': {'min': min_val, 'max': max_val},
                'u_theta': {'min': min_val, 'max': max_val},
                ... }
            state_limits_dict = {
                'x': {'min': min_val, 'max': max_val},
                'y': {'min': min_val, 'max': max_val},
                ... }
        """

        # file_path = 'config/' + file_path
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

    def __init_battlespace(self) -> None:
        """
        Creates the battlespace for the environment
        Based on the configuration file the user specifies
        this only happens if there is no battlespace already
        defined by the user
        """
        self.battlespace = BattleSpace(
            x_bounds=self.config.x_bounds,
            y_bounds=self.config.y_bounds,
            z_bounds=self.config.z_bounds,
        )

    def __init__agents(self) -> None:
        """
        Randomly spawns agents in the environment

        Right now for our case we are going to center
        this agent in the center of the battlespace with
        some perturbation
        """

        num_evaders = self.config.num_evaders
        agent_id = 0
        for i in range(num_evaders):
            rand_x = np.random.uniform(-10, 10)
            rand_y = np.random.uniform(-10, 10)
            rand_z = np.random.uniform(self.state_limits['z']['min'] + 15,
                                       self.state_limits['z']['max'] - 15)
            rand_z = 50
            rand_phi = np.random.uniform(self.state_limits['phi']['min'],
                                         self.state_limits['phi']['max'])
            rand_theta = np.random.uniform(self.state_limits['theta']['min'],
                                           self.state_limits['theta']['max'])
            rand_psi = np.random.uniform(0, 2 * np.pi)
            rand_velocity = np.random.uniform(
                self.state_limits['v']['min'] + 5, self.state_limits['v']['max'] - 5)
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
            radius_bubble = self.config.bubble_radius
            evader: Evader = Pursuer(
                battle_space=self.battlespace,
                state_vector=state_vector,
                sim_interface=sim_interface,
                radius_bubble=radius_bubble,
                is_controlled=True,
                id=agent_id)
            self.insert_agent(evader)
            agent_id += 1

    def __init_fdm(self) -> None:
        """
        This environment is used to initialize the FDM for all agents,
        which is JSBSim in this case
        """
        if self.get_all_agents is None:
            raise ValueError("all_agents is not initialized!")
        for agent in self.battlespace.all_agents:
            agent.sim_interface.sim.init_fdm()

    def __init_target(self) -> None:
        self.randomize_target: bool = self.config.randomize_target
        if self.randomize_target:
            min_radius: float = self.config.target_spawn_radius_min
            max_radius: float = self.config.target_spawn_radius_max

            # random heading
            rand_heading = np.random.uniform(0, 2 * np.pi)

            # get the agent position
            agent: Agent = self.get_controlled_agents[0]

            target_x = agent.state_vector.x + \
                np.random.uniform(min_radius, max_radius)*np.cos(rand_heading)

            target_y = agent.state_vector.y + \
                np.random.uniform(min_radius, max_radius)*np.sin(rand_heading)

            # so this is confusing but the way we spawn the target
            # is going to be based on
            target_z = np.random.uniform(self.state_limits['z']['min'] + 15,
                                         self.state_limits['z']['max'] - 15)

            self.target = StateVector(x=target_x, y=target_y, z=target_z,
                                      roll_rad=0, pitch_rad=0, yaw_rad=0, speed=0)

        else:
            self.target = StateVector(x=self.config.target_x,
                                      y=self.config.target_y,
                                      z=self.config.target_z,
                                      roll_rad=0, pitch_rad=0, yaw_rad=0, speed=0)

    def get_roll_mask(self, agent: Agent,
                      margin: float = 10) -> np.ndarray:
        """
        Computes a binary mask for the roll commands based on the aircraft's current position
        relative to the environment boundaries. The mask is applied only when the aircraft is
        nearing the x or y limits. If the aircraft is close to a boundary, only those roll commands
        that steer it back toward the center are allowed. Additionally, all roll commands that would
        result in a roll outside the allowed roll limits are masked out.

        Args:
            agent (Agent): The agent whose roll commands are to be masked. It is assumed that
                        agent.state_vector contains the current x, y, yaw_rad (heading), and roll_rad.
            margin (float, optional): The distance (meters) from the boundary at which the masking starts

        Returns:
            np.ndarray: A binary mask array (with the same length as self.roll_commands) where a 1 indicates
                        that the corresponding roll command is valid, and 0 indicates it is masked.
        """
        # Extract current state information.
        current_roll = agent.state_vector.roll_rad
        current_heading = agent.state_vector.yaw_rad
        current_x = agent.state_vector.x
        current_y = agent.state_vector.y

        # Get the environment boundaries from the configuration.
        x_min, x_max = self.config.x_bounds
        y_min, y_max = self.config.y_bounds

        # Check whether the aircraft is near any boundary.
        near_boundary = (
            (current_x - x_min < margin) or (x_max - current_x < margin) or
            (current_y - y_min < margin) or (y_max - current_y < margin)
        )

        # Start with an all-ones mask (i.e. all actions allowed).
        mask = np.ones_like(self.roll_commands, dtype=np.int8)
        # If near a boundary, compute a desired heading that steers the aircraft back to the center.
        # if near_boundary:
        #     center_x = (x_min + x_max) / 2.0
        #     center_y = (y_min + y_max) / 2.0
        #     desired_heading = np.arctan2(
        #         center_y - current_y, center_x - current_x)

        #     # Compute the heading error and normalize to [-pi, pi].
        #     heading_error = desired_heading - current_heading
        #     heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        #     # For each roll command, disable those that do not steer the heading toward the center.
        #     for i, roll_delta in enumerate(self.roll_commands):
        #         # Assume that a positive roll_delta tends to turn the aircraft right (increasing heading)
        #         # and a negative roll_delta tends to turn left (decreasing heading).
        #         if heading_error > 0 and roll_delta <= 0:
        #             mask[i] = 0
        #         elif heading_error < 0 and roll_delta >= 0:
        #             mask[i] = 0
        #         elif np.abs(heading_error) < np.deg2rad(5):
        #             # When nearly aligned, allow only very small roll adjustments.
        #             if np.abs(roll_delta) > np.deg2rad(5):
        #                 mask[i] = 0

        # Always enforce the aircraft's roll limits regardless of boundary proximity.
        roll_min = self.control_limits['u_phi']['min']
        roll_max = self.control_limits['u_phi']['max']
        for i, roll_delta in enumerate(self.roll_commands):
            # new_roll = current_roll + roll_delta
            new_roll = roll_delta
            if new_roll < roll_min or new_roll > roll_max:
                mask[i] = 0

        return mask

    def observe(self) -> gym.spaces.Dict:
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
        ego_agent: Agent = self.get_controlled_agents[0]
        deltas: np.array = self.target - ego_agent.state_vector
        dx = self.target.x - ego_agent.state_vector.x
        dy = self.target.y - ego_agent.state_vector.y
        dz = self.target.z - ego_agent.state_vector.z

        obs = [ego_agent.state_vector.x,
               ego_agent.state_vector.y,
               ego_agent.state_vector.z,
               ego_agent.state_vector.roll_rad,
               ego_agent.state_vector.pitch_rad,
               ego_agent.state_vector.yaw_rad,
               ego_agent.state_vector.speed,
               dx,
               dy,
               dz]

        obs = np.array(obs, dtype=np.float32)

        # check observaition between low and high
        low = self.observation_space["observations"].low
        high = self.observation_space["observations"].high

        if obs.shape != low.shape or obs.shape != high.shape:
            raise ValueError("Values not the same shape recieved and expected",
                             obs.shape, low.shape)

        # if not we want to print the shape of the obs, low and high
        # print(f"Observation shape: {obs.shape}, low shape: {low.shape}, high shape: {high.shape}")
        # oth
        # clip the airspeed
        speed_idx: int = ObservationIndex.AIRSPEED.value
        obs[speed_idx] = np.clip(
            obs[speed_idx], low[speed_idx], high[speed_idx])

        # we're goign to clip the relative position to the target
        obs[4] = np.clip(obs[4], low[4], high[4])
        obs[7] = np.clip(obs[7], low[7], high[7])
        obs[8] = np.clip(obs[8], low[8], high[8])
        obs[9] = np.clip(obs[9], low[9], high[9])
        if np.any(obs < low) or np.any(obs > high):
            # print the one out of bounds
            for i, (obs_val, low_val, high_val) in enumerate(zip(obs, low, high)):
                if obs_val < low_val or obs_val > high_val:
                    raise ValueError("Observation out of bounds",
                                     f"Observation {i} out of bounds: {obs_val} not in [{low_val}, {high_val}]")

        action_mask: np.array = self.get_action_mask()

        return {"observations": obs, "action_mask": action_mask}

    def get_action_mask(self) -> np.ndarray:
        """
        Compute the action mask for the controlled agent.

        Returns:
            np.ndarray: A binary mask array for each discrete action dimension.
                        The output shape is (num_action_dimensions, ),
                        where each element is an array of 0s and 1s for that dimension.
        """
        # For simplicity, assume one controlled agent.
        agent = self.get_controlled_agents[0]

        # Create default masks (all ones) for each action dimension.
        roll_mask = np.ones(len(self.roll_commands), dtype=np.int8)
        alt_mask = np.ones(len(self.alt_commands), dtype=np.int8)
        vel_mask = np.ones(len(self.airspeed_commands), dtype=np.int8)

        # Example: For altitude, ensure that a commanded altitude doesn’t exceed state limits.
        # current_alt = agent.state_vector.z
        alt_max = self.state_limits['z']['max']
        alt_min = self.state_limits['z']['min']
        current_alt = agent.state_vector.z

        # let's mask this based on max and min rates of climb
        climb_rate = 10  # m/s
        descent_rate = 20  # m/s
        alt_max = current_alt + climb_rate
        alt_min = current_alt - descent_rate
        # clip it to the bounds
        alt_max = min(alt_max, self.state_limits['z']['max'])
        alt_min = max(alt_min, self.state_limits['z']['min'])
        # Loop over altitude command options (assuming these commands represent absolute altitude or a delta)
        for i, alt_cmd in enumerate(self.alt_commands):
            # Adjust this condition based on whether alt_cmd is an absolute value or an increment.
            if (alt_cmd > alt_max) or (alt_cmd < alt_min):
                alt_mask[i] = 0

        # Example: For roll, check if the command is within the aircraft's roll limits.
        roll_mask = self.get_roll_mask(agent)

        # Example: For velocity, enforce the velocity limits.
        # current_vel = agent.state_vector.speed
        # TODO: Mask if close too stalling or max speed
        vel_max = self.control_limits['v_cmd']['max']
        vel_min = self.control_limits['v_cmd']['min']
        for i, vel_cmd in enumerate(self.airspeed_commands):
            if vel_cmd < vel_min or vel_cmd > vel_max:
                vel_mask[i] = 0

        # Combine the masks into one 3D array.
        # The overall mask at (i,j,k) is 1 only if roll_mask[i], alt_mask[j], and vel_mask[k] are all 1.

        full_mask = np.concatenate([roll_mask, alt_mask, vel_mask])
#
        # assert the size
        assert full_mask.size == self.action_space.nvec.sum()

        return full_mask

    def simulate(self, action_dict: np.ndarray, use_multi: bool = False) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step()

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        This method needs to be implemented by the child class
        """
        if seed is not None:
            np.random.seed(seed)
            # random.seed(seed)

        # Clear any previous state
        self.battlespace.clear_jsbsim()
        self.battlespace.clear_agents()

        # Reinitialize agents and then the FDM
        self.__init_battlespace()
        self.__init__agents()
        self.__init_fdm()  # Ensure FDM is initialized only after agents exist
        self.__init_target()

        self.current_step: int = 0
        observation = self.observe()
        infos = {}
        return observation, infos

    def is_out_of_bounds(self, state_vector: StateVector) -> bool:
        """
        Check if the state vector is out of bounds.
        """
        x, y, z = state_vector.x, state_vector.y, state_vector.z
        return (x < self.config.x_bounds[0] or x > self.config.x_bounds[1] or
                y < self.config.y_bounds[0] or y > self.config.y_bounds[1] or
                z < self.config.z_bounds[0] or z > self.config.z_bounds[1])

    def compute_intermediate_reward(self) -> float:
        """
        Compute the intermediate reward for the agent based on the current state.

        Args:
            state_vector (StateVector): The current state of the agent.

        Returns:
            float: The intermediate reward for the agent.
        """
        # Compute the distance to the target.
        ego_agent: Agent = self.get_controlled_agents[0]
        state_vector: StateVector = ego_agent.state_vector
        distance = self.target.distance_3D(state_vector)

        # Compute the dot product between the agent's velocity vector and the vector to the target.
        dot_product = self.target.dot_product_2D(state_vector)
        # get the heading of the agent
        dx = self.target.x - state_vector.x
        dy = self.target.y - state_vector.y

        # get the heading of the agent
        heading = np.arctan2(dy, dx)
        # get the heading error
        heading_error = heading - state_vector.yaw_rad

        # get the heading error
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # Compute the reward based on the change in distance and dot product.
        # we want the agent to get closer to the target
        # so if the distance is decreasing we give a positive reward
        # ie old distance was 60, new distance is 50, reward = +10
        # reward = (self.old_distance_from_target - distance)
        reward = 0.0

        if distance <= 150:
            dz = self.target.z - state_vector.z
            # penalize for dz error
            reward -= 0.1*np.abs(dz)

        reward -= 0.1 * np.abs(heading_error)

        # penalize for line of sight
        # if the agent is heading away from the target
        # we give a negative reward

        # we want the agent to be heading towards the target
        # so if the dot product is increasing we give a positive reward
        # ie old dot product was 0.5, new dot product is 0.7, reward = +0.2
        # reward += 0.1 * (dot_product - self.old_dot_product)

        # Update the stored values for the next step.
        self.old_distance_from_target = distance
        self.old_dot_product = dot_product

        return reward

    def get_altitude_mask(self) -> np.ndarray:
        obs: Dict = self.observe()
        action_mask: np.ndarray = obs["action_mask"]
        alt_mask: np.ndarray = action_mask[len(self.roll_commands):len(
            self.roll_commands) + len(self.alt_commands)]
        return alt_mask

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        This method needs to be implemented by the child class
        """
        # self.battlespace.act(action)
        # self.battlespace.step()
        self.current_step += 1

        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False

        if self.ctrl_counter % self.ctrl_time_index == 0 and self.ctrl_counter != 0 \
                or self.old_action is None:
            action: np.ndarray = self.discrete_to_continuous_action(action)
            self.simulate(action, use_multi=False)
            self.ctrl_counter = 0
            self.old_action: np.ndarray = action
        else:
            self.ctrl_counter += 1
            self.simulate(self.old_action, use_multi=False)

        distance_from_target: float = self.target.distance_3D(
            self.get_controlled_agents[0].state_vector)
        capture_radius: float = self.config.capture_radius
        observation = self.observe()

        # alt_mask = self.get_altitude_mask()

        infos = {
            "distance_from_target": distance_from_target,
            "old_distance_from_target": self.old_distance_from_target,
            "dot_product": self.target.dot_product_2D(
                self.get_controlled_agents[0].state_vector),
            "old_dot_product": self.old_dot_product,
            "agent_state": self.get_controlled_agents[0].state_vector
        }
        if self.current_step >= self.config.num_env_steps:
            print("ran out of time")
            terminated = True
        elif self.is_out_of_bounds(self.get_controlled_agents[0].state_vector):
            print("Agent is out of bounds",
                  self.get_controlled_agents[0].state_vector)
            print("bounds: ", self.config.x_bounds,
                  self.config.y_bounds, self.config.z_bounds)
            terminated = True
            reward = -self.terminal_reward
        elif distance_from_target <= capture_radius:
            print("Agent has captured the target")
            terminated = True
            reward = self.terminal_reward
        else:
            reward = self.compute_intermediate_reward()

        self.current_step += 1
        return observation, reward, terminated, truncated, infos
