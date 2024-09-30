from typing import Dict, Text
from typing import Dict, List, Optional, Text, Tuple, TypeVar
import numpy as np
import gymnasium 
import copy
import random
import pickle as pkl

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd

from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from jarvis.envs.battle_space_2d import BattleSpace
from jarvis.config import env_config_2d as env_config
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane2D import Agent, Evader, Pursuer, Obstacle
from jarvis.utils.math import normalize_obs, unnormalize_obs
# from jarvis.assets.BaseObject2D import BaseObject, Radar, RadarSystem
#from jarvis.assets.Radar2D import Radar2D as Radar
from jarvis.assets.Radar2D import RadarSystem2D as RadarSystem, RadarParameters
from jarvis.assets.Radar2D import Radar2D as Radar

# this is a simple environment that will be used to test the RL algorithms
"""
I might need to abstract all the annoying stuff like the action and observation spaces
using inheritance from the gymnasium.Env class. Then 
have each policy environment inherit from this class. So I can add policies
such as :
- EvaderPolicy
- TargetPolicy
"""

class AbstractBattleEnv(gymnasium.Env):
    """
    Use this as a template?
    - Should have basic observations and then you can add onto it
    It should only have ego agent observations then you can add onto it
    
    """
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 vec_env:VecNormalize=None,
                 use_discrete_actions:bool=False) -> None:
        self.config = self.default_config()
        self.dt = env_config.DT
        self.time_steps = env_config.TIME_STEPS
        self.current_step = 0
        self.config = self.default_config()
        self.use_stable_baselines = use_stable_baselines        
        self.state = None
        self.done = False
        self.upload_norm_obs = upload_norm_obs
        self.vec_env = vec_env
        self.use_discrete_actions = use_discrete_actions
        self.agents = agents
        self.battlespace:BattleSpace = battlespace
    
    @classmethod
    def default_config(cls) -> Dict:
        config = {}
        config.update({
            "x_bounds": env_config.X_BOUNDS,
            "y_bounds": env_config.Y_BOUNDS,
            "z_bounds": env_config.Z_BOUNDS,
            "num_evaders": env_config.NUM_AGENTS,
            "num_pursuers": env_config.NUM_PURSUERS,
            "use_pursuer_heuristics": env_config.USE_PURSUER_HEURISTICS,
            "dt": env_config.DT,
            "ai_pursuers": env_config.AI_PURSUERS,
            "bubble_radius": 5,
            "capture_radius": env_config.CAPTURE_RADIUS,
            #this is for pursuers
            "min_spawn_distance": env_config.MIN_SPAWN_DISTANCE,
            "max_spawn_distance": env_config.MAX_SPAWN_DISTANCE
        })
        
        return config
    
    @property
    def vehicle(self) -> Agent:
        """First (default) controlled vehicle."""
        return self.agents[0] \
            if self.agents else None
    
    def __init_battlespace(self) -> BattleSpace:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")
            
    def __init_agents(self) -> List[Agent]:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")

    def simulate(self, action_dict:np.ndarray, use_multi:bool=True) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step(self.dt)

    def get_current_observation(self, agent_id:int) -> np.ndarray:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")
       
    def get_agent_observation_space(self, agent_id:int) -> spaces.Box:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")

    def get_agent_action_space(self, agent_id:int) -> spaces.Box:
        """
        Assumes that the action space for all agents are the same
        """
        #check if self.battlespace is not None
        if self.battlespace is None:
            raise ValueError("Battlespace is None")
        
        #check if agents is not None
        if self.battlespace.agents is None:
            raise ValueError("Agents is None")
        
        high_action = []
        low_action = []
        agent : Agent = self.battlespace.agents[agent_id]
        if agent.is_pursuer:
            action_config = env_config.pursuer_control_constraints
            high_action, low_action = self.map_config(action_config)
        else:
            action_config = env_config.evader_control_constraints
            high_action, low_action = self.map_config(action_config)
            
        return spaces.Box(low=np.array(low_action), 
                          high=np.array(high_action), 
                          dtype=np.float32)

    def init_action_spaces(self, use_stable_baselines:bool) -> spaces.Dict:
        """
        Assumes that you have the same action space for all agents
        If user instantiates the environment with use_discrete_actions=True
        then the action space will be a MultiDiscrete space
        """
        if self.battlespace is None:
            raise ValueError("Battlespace is None")
        
        #check if agents is not None
        if self.battlespace.agents is None:
            raise ValueError("Agents is None")
        
        # action_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            if agent.is_controlled:
                
                action_space = self.get_agent_action_space(
                    agent_id=agent.id)
                
                if self.use_discrete_actions:
                    
                    # self.heading_commands = np.arange(
                    #     action_space.low[0], action_space.high[0], np.deg2rad(5))
                    
                    self.heading_commands = np.linspace(
                        action_space.low[0], action_space.high[0], 19)
                    
                    # self.velocity_commands = np.arange(
                    #     action_space.low[1], action_space.high[1], 1)
                    
                    self.velocity_commands = np.linspace(
                        action_space.low[1], action_space.high[1], 10)
                    
                    self.len_velocity_commands = len(self.velocity_commands)
                    self.len_heading_commands = len(self.heading_commands)
                    
                    print("length of heading commands: ", self.len_heading_commands)
                    print("length of velocity commands: ", self.len_velocity_commands)
                    
                    action_space = spaces.MultiDiscrete(
                        [self.len_heading_commands, self.len_velocity_commands])
                    return action_space
                
                elif use_stable_baselines:
                    norm_action_space = spaces.Box(
                        low=-1.0, high=1.0, shape=action_space.shape, 
                        dtype=np.float32)
                    return norm_action_space
                else: 
                    return self.get_agent_action_space(agent_id=agent.id)
            
            
    def init_observation_spaces(self) -> spaces.Dict:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")
        
    def map_config(self, action_config:Dict) -> Tuple[List,List]:
        high = []
        low = []
        for k,v in action_config.items():
            if 'max' in k:
                high.append(v)
            elif 'min' in k:
                low.append(v)

        return high, low

    def denormalize_action(self, action: np.ndarray, agent_id: int) -> np.ndarray:
        """
        This assumes the action space for all environments are the same
        """
        #check if self.battlespace is not None
        if self.battlespace is None:
            raise ValueError("Battlespace is None")
        
        agent: Agent = self.battlespace.agents[agent_id]
        action_space = self.get_agent_action_space(agent_id)
        low = action_space.low
        high = action_space.high
        # Scale the action from [-1, 1] to the original action space range
        action = low + (0.5 * (action + 1.0) * (high - low))
        return np.clip(action, low, high)
    
    def return_action_dict(self, action:np.ndarray) -> Dict:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")
        
    def return_observation_dict(self, observation:np.ndarray) -> Dict:
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")

    def reset(self, *, seed=None, options=None):
        """
        This method needs to be implemented by the child class
        """
        raise NotImplementedError("This method needs to be \
            implemented by the child class")

class RCSEnv(AbstractBattleEnv):
    """
    Requires the agent to have an RCS table or function to lookup RCS values
    """
    def __init__(self, 
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 vec_env:VecNormalize=None,
                 use_own_target:bool=False,
                 target:Obstacle=None,
                 use_discrete_actions:bool=False) -> None:
        super().__init__(spawn_own_space=spawn_own_space,
                            spawn_own_agents=spawn_own_agents,
                            battlespace=battlespace,
                            use_stable_baselines=use_stable_baselines,
                            agents=agents,
                            upload_norm_obs=upload_norm_obs,
                            vec_env=vec_env,
                            use_discrete_actions=use_discrete_actions)
        
        self.use_own_target = use_own_target
        self.target = target
        if not spawn_own_space and battlespace is None:
            self.battlespace = self.__init_battlespace(
                use_own_target=self.use_own_target,
                target=target)
            
        if not spawn_own_agents and agents is None:
            self.all_agents = self.__init_agents()
            self.battlespace.agents = self.all_agents
            self.agents = [agent for agent in self.all_agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
            self.target = self.make_target(spawn_own_target=self.use_own_target,
                                own_target=target)
            self.__init_radar_system()
            #self.battlespace.insert_radar_system()
        
        # radar to neutralize
        self.idx_radar = int(np.random.uniform(0, len(self.battlespace.radar_system.radars)))
        self.idx_radar = 1
        self.assigned_target = self.battlespace.radar_system.radars[self.idx_radar]
        self.action_space = self.init_action_spaces(self.use_stable_baselines)
        self.observation_space = self.init_observation_spaces()
        self.old_distance = 0.0
        self.seed = None
        self.terminateds = False
        self.truncateds = False
        self.reward = 0
        self.current_step = 0
        self.max_steps = env_config.MAX_NUM_STEPS
        
    def spawn_radars(self, target:Obstacle) -> List[Radar]:
        evader:Evader = self.agents[0]
        dx = evader.state_vector.x - target.x
        dy = evader.state_vector.y - target.y
        los = np.arctan2(dy, dx)
        
        random_spawn_distance = np.random.uniform(
            env_config.RADAR_SPAWN_MIN_DISTANCE, env_config.RADAR_SPAWN_MAX_DISTANCE)
        
        
        radar_list:List[Radar] = []
        # num_radars = env_config.NUM_RADARS
        radar_id = 0
        num_radars = np.random.uniform(env_config.NUM_RADARS_MIN, env_config.NUM_RADARS_MAX)
        num_radars = int(num_radars)
        first_radar_x = target.x + random_spawn_distance*np.cos(los)
        first_radar_y = target.y + random_spawn_distance*np.sin(los)
        first_radar_position = StateVector(
            x=first_radar_x,
            y=first_radar_y,
            z=0,
            roll_rad=0,
            pitch_rad=0,
            yaw_rad=los,
            speed=0
        )
        radar_params = RadarParameters(
            false_alarm_rate=0.1,
            position = first_radar_position,
            max_fov_dg=env_config.RADAR_FOV,
            range_m=env_config.RADAR_RANGE,
            c1 = -0.25,
            c2 = 1000
        )
        radar = Radar(radar_parameters=radar_params,
                      radar_id=radar_id)
        radar_list.append(radar)
        radar_id += 1
        
        left_over_radars = num_radars - len(radar_list)
        upper_radars = left_over_radars//2
        lower_radars = left_over_radars - upper_radars
        
        next_radar:Radar = radar
        for i in range(upper_radars):
            random_spawn_distance = np.random.uniform(
                env_config.RADAR_SPAWN_MIN_DISTANCE, env_config.RADAR_SPAWN_MAX_DISTANCE)
            dx = next_radar.upper_bound[0] - next_radar.position[0]
            dy = next_radar.upper_bound[1] - next_radar.position[1]
            theta = np.arctan2(dy, dx)
            radar_x = target.x + random_spawn_distance*np.cos(theta)
            radar_y = target.y + random_spawn_distance*np.sin(theta)
            radar_position = StateVector(
                x=radar_x,
                y=radar_y,
                z=0,
                roll_rad=0,
                pitch_rad=0,
                yaw_rad=theta,
                speed=0
            )
            radar_params = RadarParameters(
                false_alarm_rate=0.1,
                position = radar_position,
                max_fov_dg=env_config.RADAR_FOV,
                range_m=env_config.RADAR_RANGE,
                c1 = -0.25,
                c2 = 1000
            )
            radar = Radar(radar_parameters=radar_params,
                          radar_id=radar_id)
            next_radar = radar
            radar_list.append(radar)
            radar_id += 1
        
        next_radar = radar_list[0]
        for i in range(lower_radars):
            random_spawn_distance = np.random.uniform(
                env_config.RADAR_SPAWN_MIN_DISTANCE, env_config.RADAR_SPAWN_MAX_DISTANCE)
            dx = next_radar.lower_bound[0] - next_radar.position[0]
            dy = next_radar.lower_bound[1] - next_radar.position[1]
            theta = np.arctan2(dy, dx)
            radar_x = target.x + random_spawn_distance*np.cos(theta)
            radar_y = target.y + random_spawn_distance*np.sin(theta)
            radar_position = StateVector(
                x=radar_x,
                y=radar_y,
                z=0,
                roll_rad=0,
                pitch_rad=0,
                yaw_rad=theta,
                speed=0
            )
            radar_params = RadarParameters(
                false_alarm_rate=0.1,
                position = radar_position,
                max_fov_dg=env_config.RADAR_FOV,
                range_m=env_config.RADAR_RANGE,
                c1 = -0.25,
                c2 = 1000
            )
            radar = Radar(radar_parameters=radar_params,
                            radar_id=radar_id)
            next_radar = radar
            radar_list.append(radar)
            radar_id += 1
            
        return radar_list
            
    def __init_agents(self) -> None:
        agents = []
        counter = 0
        
        for i in range(self.config["num_evaders"]):
            min_speed = env_config.evader_observation_constraints['airspeed_min']
            max_speed = env_config.evader_observation_constraints['airspeed_max']
            min_spawn_distance = self.config["min_spawn_distance"]
            max_spawn_distance = self.config["max_spawn_distance"]

            rand_x = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2
            rand_y = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2   
            rand_x = np.random.uniform(-700, -701)
            rand_y = np.random.uniform(-700, -701)
            evader = Evader(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    rand_x,
                    rand_y,
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"]
            )
            agents.append(evader)
            counter += 1
            #spawn pursuers at a distance from the evader
        return agents
    

    def __init_battlespace(self, use_own_target:bool=False,
                           target:Obstacle=None) -> BattleSpace:
        # Needs to include radar systems
        battle_space =  BattleSpace(
            x_bounds=self.config["x_bounds"],
            y_bounds=self.config["y_bounds"],
            z_bounds=self.config["z_bounds"]
        )
        
        ## Update this? Need to make it guard against the 
        # target being None
        if use_own_target and target is not None: 
            target = Obstacle(
                x = np.random.uniform(self.config["x_bounds"][0], self.config["x_bounds"][1]),
                y = np.random.uniform(self.config["y_bounds"][0], self.config["y_bounds"][1]),
                z = 0,
                radius = env_config.TARGET_RADIUS
            )
        else:
            target = target
            
        battle_space.insert_target(target)
    
        return battle_space
        
    def __init_radar_system(self) -> RadarSystem:
        """
        To make the environment a bit more realistic we would 
        probably want to create a radar system 
        that interweaves the radars in the environment 
        """
        radars:List[Radar] = self.spawn_radars(self.battlespace.target)
        radar_system:RadarSystem = RadarSystem(radar_system=radars)
        self.battlespace.insert_radar_system(
            radar_system=radar_system) 
        
        # return radar_system
        
    def init_observation_spaces(self) -> spaces.Dict:
        obs_space = self.get_agent_observation_space(self.vehicle.id)
        
        return obs_space
        
    def get_agent_observation_space(self, agent_id: int) -> spaces.Box:
        """
        Order is as follows:
            x 
            y
            theta 
            
            dx to radar target
            dy to radar target
            distance to radar target
            heading to radar target
            
            dx to other radar
            dy to other radar
            distance to other radar
            heading to other radar
            last value is the rcs probability
        """
        # ego_agent = self.battlespace.agents[agent_id]
        
        obs_config = env_config.evader_observation_constraints
        high_obs, low_obs = self.map_config(obs_config)
        
        #add the radar observations
        for radar in self.battlespace.radar_system.radars:
            
            radar:Radar = radar
            low_x = env_config.X_BOUNDS[0]
            high_x = env_config.X_BOUNDS[1]
            
            low_y = env_config.Y_BOUNDS[0]
            high_y = env_config.Y_BOUNDS[1]
            
            distance_squared = (high_x - low_x)**2 + (high_y - low_y)**2
            distance = np.sqrt(distance_squared)
            
            low = [
                low_x, 
                low_y, 
                0,
                -np.pi,
            ]
            
            high = [
                high_x,
                high_y,
                distance,
                np.pi
            ]
            
            high_obs.extend(high)
            low_obs.extend(low)
            
        #add the rcs probability
        low_obs.append(0)
        high_obs.append(1)

        return spaces.Box(low=np.array(low_obs),
                            high=np.array(high_obs),
                            dtype=np.float32)
        
        
    def make_target(self, spawn_own_target:bool=False,
                    own_target: Obstacle=None) -> Obstacle:
        if spawn_own_target and own_target is None:
            self.battlespace.insert_target(
                own_target)
            return own_target
        
        factor = 2
        #rand_x = np.random.uniform(-self.config["x_bounds"][1]/factor, self.config["x_bounds"][1]/factor)
        #rand_y = np.random.uniform(-self.config["y_bounds"][1]/factor, self.config["y_bounds"][1]/factor)
        rand_x = 0
        rand_y = 0
        target = Obstacle(
            x = rand_x,
            y = rand_y,
            z = 0,
            radius = env_config.TARGET_RADIUS
        )
        self.battlespace.insert_target(target)
        return target
        
    def get_reward(self, observation:np.ndarray) -> float:
        """
        This method needs to be implemented by the child class
        """
        # reward for getting closer t
        dx = self.assigned_target.position[0] - self.vehicle.state_vector.x
        dy = self.assigned_target.position[1] - self.vehicle.state_vector.y
        distance = np.sqrt(dx**2 + dy**2)
        
        delta_distance = self.old_distance - distance
        detection_probability = observation[-1] # this value is between 0 and 1
        
        distance_weight = 2
        reward = (distance_weight*delta_distance) - detection_probability
        self.old_distance = distance
        return reward
        
    def get_current_observation(self, agent_id:int) -> np.ndarray:
        """
        Needs to return the ego observation and rcs probability?
        x 
        y
        theta 
        dx to radar target
        dy to radar target
        distance to radar target
        heading to radar target
        dx to other radar
        dy to other radar
        distance to other radar
        heading to other radar
        last value is the rcs probability
        """
        agent = self.battlespace.agents[agent_id]
        observation = agent.get_observation()
        radar_system:RadarSystem = self.battlespace.radar_system
        rcs_probability = radar_system.probability_of_detection_system(agent)

        for radar in self.battlespace.radar_system.radars:
            radar:Radar = radar
            dx = radar.position[0] - agent.state_vector.x
            dy = radar.position[1] - agent.state_vector.y
            distance = np.sqrt(dx**2 + dy**2)
            #relatitive heading from agent to radar
            heading = np.arctan2(dy, dx)
            # relative_heading = heading - agent.state_vector.yaw_rad
            # #wrap the heading
            # relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi
            incident_angle = np.deg2rad(radar_system.compute_angle_of_incidence(
                radar, agent))
            observation = np.append(observation, [dx, dy, distance, incident_angle])
        
        observation = np.append(observation, rcs_probability)
        
        if observation.shape[0] != self.observation_space.shape[0]:
            raise ValueError("Observation shape is not correct", observation.shape,
                             self.observation_space.shape[0])
            
        return observation.astype(np.float32)
      
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
                    
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        
        self.battlespace = self.__init_battlespace()
        self.all_agents = self.__init_agents()
        self.battlespace.agents = self.all_agents
        self.agents = [agent for agent in self.all_agents
                        if agent.is_controlled and not agent.is_pursuer]
        self.target = self.make_target(spawn_own_target=self.use_own_target,
                                       own_target=self.target)
        self.__init_radar_system()
        
        self.idx_radar = int(np.random.uniform(0, len(self.battlespace.radar_system.radars)))
        self.idx_radar = 1
        self.assigned_target = self.battlespace.radar_system.radars[self.idx_radar]
        
        self.current_step = 0
        self.terminateds = False
        self.truncateds = False
        self.reward = 0
        
        observation = self.get_current_observation(self.vehicle.id)
        self.old_distance = 0
        infos = {}
        
        return observation, infos
    
    def step(self, action: np.ndarray, 
             norm_action:bool=True) -> Tuple[np.ndarray, float, bool, Dict]:
        self.reward = 0
        info = {}
        
        if self.use_discrete_actions:
            action = self.heading_commands[action[0]], self.velocity_commands[action[1]]
            action = np.array(action)
        elif self.use_stable_baselines:
            action = self.denormalize_action(action, self.vehicle.id)
            
        self.simulate(action, use_multi=False)
        current_obs = self.get_current_observation(self.vehicle.id)
        self.reward = self.get_reward(current_obs)
        self.current_step += 1
        
        #check if close to target
        dx = self.assigned_target.position[0] - self.vehicle.state_vector.x
        dy = self.assigned_target.position[1] - self.vehicle.state_vector.y
        distance = np.sqrt(dx**2 + dy**2)
        detection_probability = current_obs[-1]
        info['detection_probability'] = detection_probability
        
        if distance < env_config.RADAR_CAPTURE_DISTANCE:
            self.terminateds = True
            self.truncateds = True
            self.reward = 1000
            print("captured")        
        elif self.current_step >= self.max_steps:
            self.terminateds = True
            self.truncateds = True
            self.reward = -1000    
        elif detection_probability >= 0.50:
            print("You've been detected")
            self.terminateds = True
            self.truncateds = True
            self.reward = -1000
            # print("You've been detected")
            
        # Don't normalize the environment and use discrete actions
        if self.upload_norm_obs and self.vec_env is not None:
            current_obs = normalize_obs(current_obs, self.vec_env)
            
        self.observation = current_obs
            
        return current_obs, self.reward, self.terminateds, self.truncateds, info
                      
class AvoidThreatEnv(AbstractBattleEnv):
    """
    This environment is used to train the agent to learn an avoidance policy
    from n pursuers
    """
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 use_discrete_actions:bool=False,
                 vec_env:VecNormalize=None,
                 randomize_start:bool=True,
                 randomize_threats:bool=True) -> None:
        super().__init__(spawn_own_space=spawn_own_space,
                         spawn_own_agents=spawn_own_agents,
                         battlespace=battlespace,
                         use_stable_baselines=use_stable_baselines,
                         agents=agents,
                         upload_norm_obs=upload_norm_obs,
                         use_discrete_actions=use_discrete_actions,
                         vec_env=vec_env)
        
        self.config:dict = self.default_config()
        self.dt:float = env_config.DT
        self.time_steps:int = env_config.TIME_STEPS
        self.current_step:int = 0

        self.randomize_start:bool = randomize_start
        self.randomize_threats:bool = randomize_threats
        
        if not spawn_own_space and battlespace is None:
            print("Spawning battlespace")
            self.battlespace:BattleSpace = self.__init_battlespace()
        else:
            self.battlespace = battlespace
            
        if not spawn_own_agents and agents is None:
            self.all_agents = self.__init_agents()
            self.battlespace.agents = self.all_agents
            self.agents = [agent for agent in self.all_agents
                            if agent.is_controlled and not agent.is_pursuer]
        else:
            self.all_agents = agents
            self.agents = [agent for agent in self.all_agents
                            if agent.is_controlled and not agent.is_pursuer]
            self.battlespace.agents = self.all_agents
            
        self.action_space = self.init_action_spaces(self.use_stable_baselines)
        self.observation_space = self.init_observation_spaces()
        self.seed = None
        self.terminateds = False
        self.truncateds = False
        self.reward = 0
        self.terminal_reward = 1000
        self.observation = None
        self.old_distance:float = 0.0
        
    def __init_battlespace(self) -> BattleSpace:
        return BattleSpace(
            x_bounds=self.config["x_bounds"],
            y_bounds=self.config["y_bounds"],
            z_bounds=self.config["z_bounds"]
        )
    
    def init_observation_spaces(self) -> spaces.Dict:
        # right now make this thing return a single observation space 
        
        obs_space = self.get_agent_observation_space(self.vehicle.id)
        return obs_space
        # for agent in self.battlespace.agents:
        #     agent: Agent = agent
        #     #TODO: Need to refactor since PN will be a policy, that shit's too hard
        #     if agent.is_controlled:
        #         observation_spaces[agent.id] = self.get_agent_observation_space(
        #             agent_id=agent.id
        #         )
        #         return self.get_agent_observation_space(agent_id=agent.id)
        
    def get_agent_observation_space(self, agent_id: int) -> spaces.Box:
        """
        For this avoidance of threats observation we will use the 
        following information:
        State:
            X
            Y
            Theta
            Velocity
        
        Observation:
            dx -> min(x distance from threats)
            dy -> min(y distance from threats)
            dr -> min(closest threat)
            dtheat -> the heading difference
            dvelocity -> the difference of velocity
            
        Multiply by n pursuers/threats and this will be our observation space
        """
        
        # high_obs = []
        # low_obs = []
        # agent:Agent = self.battlespace.agents[agent_id]
        ego_agent = self.vehicle

        obs_config = env_config.evader_observation_constraints
        high_obs, low_obs = self.map_config(obs_config)
        
        for agent in self.all_agents:
            agent:Agent = agent
            
            if agent.id == ego_agent.id:
                continue
                        
            low = []
            high = []

            low_x = env_config.X_BOUNDS[0]
            high_x = env_config.X_BOUNDS[1]
            
            low_y = env_config.Y_BOUNDS[0]
            high_y = env_config.Y_BOUNDS[1]
            
            distance_squared = (high_x - low_x)**2 + (high_y - low_y)**2
            min_rvelocity = -env_config.evader_observation_constraints['airspeed_max']
            max_rvelocity = env_config.evader_observation_constraints['airspeed_max']
            
            low = [low_x, low_y, 0.0, -np.pi, min_rvelocity] # min_dist, min_delta_heading
            high= [high_x, high_y, distance_squared, np.pi, max_rvelocity] # max_dist, max_delta_heading
        
            low_obs.extend(low)
            high_obs.extend(high)
            
            
        return spaces.Box(low=np.array(low_obs),
                            high=np.array(high_obs),
                            dtype=np.float32)
        
        
    def get_current_observation(self, agent_id: int) -> np.ndarray:
        """
        Returns the observation for the agent as a numpy array
        """        
        ego_agent = self.battlespace.agents[agent_id]
        current_state = self.vehicle.get_observation()
        
        observation = self.vehicle.get_observation()#np.array([])
        for agent in self.all_agents:
            agent:Agent = agent
            if agent.id == ego_agent.id:
                continue
            
            dx = agent.state_vector.x - self.vehicle.state_vector.x
            dy = agent.state_vector.y - self.vehicle.state_vector.y
            dr = np.sqrt(dx**2 + dy**2)
            dtheta = ego_agent.state_vector.heading_difference(agent.state_vector)
            dvelocity = agent.state_vector.speed - self.vehicle.state_vector.speed
            
            relative_obs = [dx, dy, dr, dtheta, dvelocity]

            observation = np.append(observation, relative_obs)

        if observation.shape[0] != self.observation_space.shape[0]:
            raise ValueError("Observation shape is not correct", 
                             observation.shape,
                             self.observation_space.shape[0])
            
        if self.upload_norm_obs and self.vec_env is not None:
            observation = normalize_obs(observation, self.vec_env)    
        
        return observation.astype(np.float32)
        
    def __init_agents(self) -> List[Agent]:
        agents = []
        counter = 0
        
        for i in range(self.config["num_evaders"]):
            min_speed = env_config.evader_observation_constraints['airspeed_min']
            max_speed = env_config.evader_observation_constraints['airspeed_max']
            # min_spawn_distance = self.config["min_spawn_distance"]
            # max_spawn_distance = self.config["max_spawn_distance"]

            #rand_x = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2
            # rand_y = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2   
            evader = Evader(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    0.0,
                    0.0,
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"]
            )
            agents.append(evader)
            counter += 1
            #spawn pursuers at a distance from the evader
        
        for i in range(self.config["num_pursuers"]):
            min_speed = env_config.pursuer_observation_constraints['airspeed_min']
            max_speed = env_config.pursuer_observation_constraints['airspeed_max']
            min_spawn = self.config["min_spawn_distance"]
            max_spawn = self.config["max_spawn_distance"]
        
            #spawn pursuers at a distance from the evader
            random_spawn_distance = np.random.uniform(min_spawn, max_spawn)
            #random value of -1 or 1
            evader: Evader = agents[0]
            random_angle = evader.state_vector.yaw_rad + np.random.uniform(-np.pi, np.pi)
            x = evader.state_vector.x + random_spawn_distance * np.cos(random_angle)
            y = evader.state_vector.y + random_spawn_distance * np.sin(random_angle)
            
            #get evader heading
            pursuer_heading = np.arctan2(evader.state_vector.y - y, 
                                         evader.state_vector.x - x)
            pursuer = pursuer_heading + np.random.uniform(-np.pi/4, np.pi/4)
            
            pursuer = Pursuer(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    x,
                    y,
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    pursuer_heading,
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"],
                capture_distance=self.config["capture_radius"]
            )
            if self.config["ai_pursuers"]:
                pursuer.is_controlled = True
            agents.append(pursuer)
            counter += 1            
        
        return agents
    
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        This method needs to be implemented by the child class
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
            
        self.battlespace = self.__init_battlespace()
        self.all_agents = self.__init_agents()
        self.battlespace.agents = self.all_agents
        self.agents = [agent for agent in self.all_agents
                        if agent.is_controlled and not agent.is_pursuer]
        self.current_step = 0
        self.terminateds = False
        self.truncateds = False
        self.reward = 0
        self.old_distance = 0.0
        
        observation = self.get_current_observation(self.vehicle.id)
        infos = {}
        return observation, infos

    def get_reward(self, observation:np.ndarray) -> float:
        """
        Reward function to avoid the pursuers
        Since the observation space is formulated as :
        [dx, dy, dr, dtheta, dvelocity]
        We want to find the minimum distance from the pursuers
        We want to find the minimum heading difference
        """
        every_pursuer_idx = 5
        rel_distance_idx = 2
        rel_theta_idx = 3
        rel_velocity_idx = 4
        
        relative_velocities = observation[rel_velocity_idx::every_pursuer_idx]
        relative_distances = observation[rel_distance_idx::every_pursuer_idx]
        relative_thetas = observation[rel_theta_idx::every_pursuer_idx]
        
        # get the minimum distance and the index
        # min_distance = np.min(relative_distances)
        # min_distance_idx = np.argmin(relative_distances)
        mean_distance = np.mean(relative_distances)
        
        # get the minimum heading difference
        min_theta = np.min(relative_thetas)
        
        reward = mean_distance - self.old_distance
        self.old_distance = mean_distance
        
        return reward
        
    def step(self, action: np.ndarray, 
             norm_action:bool=True) -> Tuple[np.ndarray, float, bool, Dict]:
        
        self.reward = 0
        info = {}
        
        if self.use_discrete_actions:
            action = self.heading_commands[action[0]], self.velocity_commands[action[1]]
            #turn to array
            action = np.array(action)  
        elif self.use_stable_baselines:
            action = self.denormalize_action(action, agent_id=0)

        self.simulate(action, use_multi=False)
        current_obs = self.get_current_observation(self.vehicle.id)
        
        self.reward = self.get_reward(current_obs)
        
        self.current_step += 1
        
        if self.current_step >= self.time_steps:
            print("Time steps exceeded")
            self.terminateds = True
            self.truncateds = True
            self.reward = self.terminal_reward
            print("terminateds", self.terminateds)
            
        
        # first let's check if our ego vehicle crashed 
        ego_agent:Evader = self.vehicle
        if ego_agent.crashed:
            self.terminateds = True
            self.truncateds = True
            self.reward = -self.terminal_reward
        
        # sift through to see if our agent made the pursuers crash into each other
        for agent in self.battlespace.agents:
            if agent.is_controlled:
                continue
            if agent.crashed and agent.is_pursuer and not ego_agent.crashed:
                print("Pursuer crashed you win", ego_agent.crashed)
                self.truncateds = True
                self.terminateds = True
                self.reward = self.terminal_reward
                
            # if agent.crashed and agent.is_pursuer: 
            #     self.truncateds = True
            #     self.terminateds = True
            #     self.reward = self.terminal_reward
                
        # Don't normalize the environment and use discrete actions
        if self.upload_norm_obs and self.vec_env is not None and not self.use_discrete_actions:
            current_obs = normalize_obs(current_obs, self.vec_env)
            
        self.observation = current_obs
        
        return current_obs, float(self.reward), self.terminateds, self.truncateds, info
            
class EngagementEnv(AbstractBattleEnv):
    """
    This is the environment that will be used to teach 
    the agent to engage on a static target
    
    We want to be within the vicinity of the target as much as possible
    Without crashing into it 
    """
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 vec_env:VecNormalize=None,
                 use_heuristic_policy:bool=False,
                 randomize_goal:bool=False,
                 randomize_start:bool=False,
                 difficulty_level:int=0,
                 use_discrete_actions:bool=False) -> None:
        super().__init__(spawn_own_space=spawn_own_space,
                         spawn_own_agents=spawn_own_agents,
                         battlespace=battlespace,
                         use_stable_baselines=use_stable_baselines,
                         agents=agents,
                         upload_norm_obs=upload_norm_obs,
                         vec_env=vec_env,
                         use_discrete_actions=use_discrete_actions)
        if not spawn_own_space and battlespace is None:
            self.battlespace = self.__init_battlespace()
            
        self.spawn_own_agents = spawn_own_agents
        self.randomize_goal = randomize_goal
        self.randomize_start = randomize_start
        self.difficulty_level = difficulty_level
        self.use_discrete_actions = use_discrete_actions
        self.iteration_check = 50
        self.current_iteration = 0
        self.lose = True
        self.success_history = []
        self.starting_random_angle = np.deg2rad(0) 
        if not spawn_own_agents and agents is None:
            self.all_agents = self.__init_agents()
            self.battlespace.agents = self.all_agents
            self.agents = [agent for agent in self.all_agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
        else:
            self.all_agents = agents
            # we awnt to control the evaders
            self.agents = [agent for agent in self.all_agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
            self.battlespace.agents = self.all_agents
        
        self.make_target()
        self.action_space = self.init_action_spaces(self.use_stable_baselines)
        self.observation_space = self.init_observation_spaces()
        self.terminateds = False
        self.truncateds = False
        self.observation = None
        self.reward = 0
        self.old_distance_to_target = None
        self.time_steps = env_config.TARGET_TIME_STEPS
        self.effector_range = env_config.EFFECTOR_RANGE
        self.use_heuristic_policy = use_heuristic_policy

    def __init_battlespace(self) -> BattleSpace:
        return BattleSpace(
            x_bounds=self.config["x_bounds"],
            y_bounds=self.config["y_bounds"],
            z_bounds=self.config["z_bounds"]
        )
        
    def make_target(self, spawn_own_target:bool=False,
                    own_target:Obstacle=None) -> None:
        """
        #TODO: test if this works
        """
        if spawn_own_target and own_target is not None:
            self.battlespace.insert_target(own_target)
            target = self.battlespace.target
            self.old_distance_to_target = target.state_vector.distance_2D(
                self.agents[0].state_vector)
            return
            
        controlled_agent: Evader = self.agents[0]
        agent_x = controlled_agent.state_vector.x
        agent_y = controlled_agent.state_vector.y
        agent_heading = controlled_agent.state_vector.yaw_rad
        
        min_spawn_distance = env_config.MIN_TARGET_DISTANCE
        max_spawn_distance = env_config.MAX_TARGET_DISTANCE
        
        #random_heading = np.random.uniform(-np.pi, np.pi)
        random_distance = np.random.uniform(min_spawn_distance, max_spawn_distance)
        random_heading = controlled_agent.state_vector.yaw_rad
        
        if self.randomize_goal:
            if self.difficulty_level == 1:
                #original_distance = np.sqrt((env_config.TARGET_X - agent_x)**2 + (env_config.TARGET_Y - agent_y)**2)
                original_angle = np.arctan2(env_config.TARGET_Y - agent_y, env_config.TARGET_X - agent_x)
                target_x = random_distance * np.cos(original_angle)
                target_y = random_distance * np.sin(original_angle)
            elif self.difficulty_level == 2:
                target_heading  = np.random.uniform(-self.starting_random_angle, 
                                                    self.starting_random_angle)
                target_x = random_distance * np.cos(target_heading)
                target_y = random_distance * np.sin(target_heading)
        else:
            target_x = env_config.TARGET_X
            target_y = env_config.TARGET_Y
            
        obstacle = Obstacle(
            x = target_x,
            y = target_y,
            z = 0,
            radius = env_config.TARGET_RADIUS
        )
        
        self.battlespace.insert_target(obstacle)
        
        target = self.battlespace.target
        self.old_distance_to_target = target.state_vector.distance_2D(
            controlled_agent.state_vector)
        
    def __init_agents(self) -> None:
        agents = []
        counter = 0
        
        for i in range(self.config["num_evaders"]):
            min_speed = env_config.evader_observation_constraints['airspeed_min']
            max_speed = env_config.evader_observation_constraints['airspeed_max']
            min_spawn_distance = self.config["min_spawn_distance"]
            max_spawn_distance = self.config["max_spawn_distance"]

            rand_x = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2
            rand_y = np.random.uniform(-min_spawn_distance, min_spawn_distance)/2   
            evader = Evader(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    0.0,
                    0.0,
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"]
            )
            agents.append(evader)
            counter += 1
            #spawn pursuers at a distance from the evader
        return agents
    
    def init_observation_spaces(self) -> spaces.Dict:
        observation_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            #TODO: Need to refactor since PN will be a policy, that shit's too hard
            if agent.is_controlled:
                observation_spaces[agent.id] = self.get_agent_observation_space(
                    agent_id=agent.id
                )
                return self.get_agent_observation_space(agent_id=agent.id)
    
    def get_agent_observation_space(self, agent_id: int) -> spaces.Box:
        """
        For this observation space we want to consider the ego
        vehicle and the target location 
        - Maybe include closest obstacles later on 
        """
        
        #check if the battlespace has a target
        if self.battlespace.target is None:
            raise ValueError("Battlespace target is None")
        
        high_obs = []
        low_obs =  []
        agent: Agent = self.battlespace.agents[agent_id]
        # for now we only want to use the evader
        if not agent.is_pursuer:
            obs_config = env_config.evader_observation_constraints
            high_obs, low_obs = self.map_config(obs_config)
            
            #insert the target location
            target = self.battlespace.target
            low_x = env_config.X_BOUNDS[0]
            high_x = env_config.X_BOUNDS[1]
            low_y = env_config.Y_BOUNDS[0]
            high_y = env_config.Y_BOUNDS[1]
            
            distance_squared = (high_x - low_x)**2 \
                + (high_y - low_y)**2
            low = [0, -np.pi]
            high = [distance_squared, np.pi]
            
            low_obs.extend(low)
            high_obs.extend(high)
            
        return spaces.Box(low=np.array(low_obs),
                            high=np.array(high_obs),
                            dtype=np.float32)
        
    def get_current_observation(self, agent_id: int) -> np.ndarray:
        """
        Return the current observation for the agent 
        should include the target location
        """
        agent = self.battlespace.agents[agent_id]
        observation = agent.get_observation()
        target = self.battlespace.target

        distance = target.state_vector.distance_2D(agent.state_vector)
        relative_heading = target.state_vector.heading_difference(agent.state_vector)
        observation = np.append(observation, [distance, relative_heading])

        #let's replace the x and y with dx and dy
        dx = target.state_vector.x - agent.state_vector.x
        dy = target.state_vector.y - agent.state_vector.y
        
        observation[0] = dx
        observation[1] = dy
        
        #clip velocity
        vel_min = env_config.evader_observation_constraints['airspeed_min']
        vel_max = env_config.evader_observation_constraints['airspeed_max']
        observation[3] = np.clip(observation[3], vel_min, vel_max)
        
        if observation.shape[0] != self.observation_space.shape[0]:
            raise ValueError("Observation shape is not correct", observation.shape,
                             self.observation_space.shape[0])
            
        if self.upload_norm_obs and self.vec_env is not None:
            observation = normalize_obs(observation, self.vec_env)    
        
        return observation.astype(np.float32)
    
    def get_target_observation(self, agent_id:int) -> np.ndarray:
        """
        Get the observation for the target
        """
        agent = self.battlespace.agents[agent_id]
        target = self.battlespace.target
        distance = target.state_vector.distance_2D(agent.state_vector)
        relative_heading = target.state_vector.heading_difference(agent.state_vector)
        return np.array([distance, relative_heading], dtype=np.float32)
    
    def reset(self, *, seed=None, options=None, agents:List[Agent]=None) -> np.ndarray:
        self.battlespace = self.__init_battlespace()
        
        if not self.spawn_own_agents and agents is None:
            self.all_agents  = self.__init_agents()
            self.battlespace.agents = self.all_agents
            self.agents = [agent for agent in self.all_agents 
                                if agent.is_controlled and 
                                not agent.is_pursuer]
        else:
            self.all_agents = agents
            self.agents = [agent for agent in self.all_agents
                                if agent.is_controlled and 
                                not agent.is_pursuer]
            self.battlespace.agents = self.all_agents

        self.make_target()
        self.current_step = 0
        self.agents = [agent for agent in self.all_agents 
                                    if agent.is_controlled and 
                                    not agent.is_pursuer]
        
        self.terminateds = False
        self.truncateds = False
        infos = {}
        
        self.current_iteration += 1
        self.curriculm_learning()
        self.lose = True        
        observation = self.get_current_observation(agent_id=0)
        return observation, infos
    
    def curriculm_learning(self) -> None:
        """
        Based on our success rate we want to increase the difficulty
        by increasing the spawn angle
        """
        if self.lose:
            value = 0
        else:
            value = 1
            
        self.success_history.append(value)

        if self.current_iteration !=0 and \
            self.current_iteration % self.iteration_check == 0:
            success_rate = np.mean(self.success_history)
            #write in a text file the angle and the success rate
            with open("success_rate.txt", "a") as f:
                #write in new line
                f.write("\n")
                f.write(f"Success rate: {success_rate}, Angle: {self.starting_random_angle}")
            
            if success_rate >= 0.9:
                if self.starting_random_angle >= np.deg2rad(180):
                    self.starting_random_angle = np.deg2rad(180)
                    print("Maximum difficulty reached")
                    
                self.starting_random_angle = self.starting_random_angle + np.deg2rad(5)
                print("Increasing difficulty new angle is: ", self.starting_random_angle)
            
            print("Resetting the success history")
            self.success_history = []
            self.current_iteration = 0
    
    def get_reward(self, observation:np.ndarray) -> float:
        """
        This method needs to be implemented by the child class
        """
        #reward for being close to the target
        target = self.battlespace.target
        agent = self.agents[0]
        distance = target.state_vector.distance_2D(agent.state_vector)
        if self.old_distance_to_target is None:
            self.old_distance_to_target = distance
            return 0
        delta_distance = self.old_distance_to_target - distance
        dx = target.state_vector.x - agent.state_vector.x
        dy = target.state_vector.y - agent.state_vector.y
        los = np.arctan2(dy, dx)
        error_los = los - agent.state_vector.yaw_rad
        # reward for being closer to target so if delta_distance is positive then
        # we should reward the agent
        heading_error = np.abs(error_los)
        self.old_distance_to_target = distance
        
        ego_unit_vector = np.array([np.cos(agent.state_vector.yaw_rad),
                                    np.sin(agent.state_vector.yaw_rad)])
        
        #target_unit_vector = np.array([np.cos(los), np.sin(los)])
        los_unit_vector = np.array([np.cos(los), np.sin(los)])
        
        dot_product = np.dot(ego_unit_vector, los_unit_vector)
        
        #TODO: make this easier to understand we want to penalize the agent for
        # being too far from the target
        #return delta_distance
        return delta_distance
    
    def heuristic_policy(self) -> np.ndarray:
        """
        This is a simple heuristic policy that will be used to 
        test the environment
        """
        agent = self.agents[0]
        target = self.battlespace.target
        dx = target.state_vector.x - agent.state_vector.x
        dy = target.state_vector.y - agent.state_vector.y
        theta = np.arctan2(dy, dx)
        heading_error  = theta - agent.state_vector.yaw_rad
        action = np.array([heading_error, 20])
        return action
    
    def is_close_to_target(self, agent:Agent) -> bool:
        target = self.battlespace.target
        distance = target.state_vector.distance_2D(agent.state_vector)
        if distance <= env_config.TARGET_RADIUS+agent.radius_bubble:
            return True
        
        return False
    
    def step(self, action:np.ndarray,
             norm_action:bool=True) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        
        self.reward = 0
        info = {}
        info['engaged'] = False
        
        if self.use_discrete_actions:
            action = self.heading_commands[action[0]], self.velocity_commands[action[1]]
            #turn to array
            action = np.array(action)  
        elif self.use_stable_baselines:
            action = self.denormalize_action(action, agent_id=0)
        
        if self.use_heuristic_policy:
            action = self.heuristic_policy()
            
        self.simulate(action, use_multi=False)
        self.observation = self.get_current_observation(agent_id=0)
        
        controlled_agent: Evader = self.agents[0]
        self.reward = self.get_reward(self.observation)
    
        #check distance to target
        target = self.battlespace.target
        distance = target.state_vector.distance_2D(controlled_agent.state_vector)
        
        #check if crashed 
        for agent in self.battlespace.agents:
            if agent.crashed:
                self.terminateds = True
                self.reward -= 100
                break
        
        if distance <= env_config.TARGET_RADIUS+controlled_agent.radius_bubble:
            print("Target reached")
            info['engaged'] = True
            self.terminateds = True
            self.reward += 100
            self.lose = False
        
        if self.current_step >= self.time_steps:
            print("Time steps exceeded")
            self.truncateds = True
            self.terminateds = True
            self.reward -= 100
    
        self.current_step += 1
        #penalize the agent for taking too long
        self.reward -= (1*0.01) 
        
        return self.observation, self.reward, self.terminateds, self.truncateds, info
    
class ThreatAvoidEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 vec_env:VecNormalize=None) -> None:
        
        self.config = self.default_config()
        self.dt = env_config.DT
        self.time_steps = env_config.TIME_STEPS
        self.current_step = 0
        self.terminal_reward = 100
        self.use_stable_baselines = use_stable_baselines        
        self.state = None
        self.done = False
        self.control_freq = 1 #(1/control_freq) is the control frequency
        self.old_action = None

        self.current_sim_num = 0
        self.num_wins = 0
        self.mean_episode_length = 0
        
        self.upload_norm_obs = upload_norm_obs
        self.vec_env = vec_env
        
        #TODO: Refactor all of this as a method
        if not spawn_own_space and battlespace is None:
            self.battlespace = self.__init_battlespace()
        else:
            self.battlespace = battlespace
        
        if not spawn_own_agents and agents is None:
            self.all_agents = self.__init_agents()
            self.battlespace.agents = self.all_agents
            self.agents = [agent for agent in self.all_agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
        else:
            self.all_agents = agents
            # we awnt to control the evaders
            self.agents = [agent for agent in self.all_agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
            self.battlespace.agents = self.all_agents
            
        self.action_space = self.init_action_spaces(self.use_stable_baselines)    
        self.observation_space = self.init_observation_spaces()
        self.terminateds = False
        self.truncateds = False
        self.observation = None
        self.reward = 0
        #pursuer position

    @property
    def vehicle(self) -> Agent:
        """First (default) controlled vehicle."""
        return self.agents[0] \
            if self.agents else None
        
    @classmethod
    def default_config(cls) -> Dict:
        config = {}
        config.update({
            "x_bounds": env_config.X_BOUNDS,
            "y_bounds": env_config.Y_BOUNDS,
            "z_bounds": env_config.Z_BOUNDS,
            "num_evaders": env_config.NUM_AGENTS,
            "num_pursuers": env_config.NUM_PURSUERS,
            "use_pursuer_heuristics": env_config.USE_PURSUER_HEURISTICS,
            "dt": env_config.DT,
            "ai_pursuers": env_config.AI_PURSUERS,
            "bubble_radius": 5,
            "capture_radius": env_config.CAPTURE_RADIUS,
            "min_spawn_distance": env_config.MIN_SPAWN_DISTANCE,
            "max_spawn_distance": env_config.MAX_SPAWN_DISTANCE
        })
        
        return config
        
    def __init_battlespace(self) -> BattleSpace:
        return BattleSpace(
            x_bounds=self.config["x_bounds"],
            y_bounds=self.config["y_bounds"],
            z_bounds=self.config["z_bounds"]
        )
        
    def __init_agents(self) -> None:
        agents = []
        counter = 0
        for i in range(self.config["num_evaders"]):
            min_speed = env_config.evader_observation_constraints['airspeed_min']
            max_speed = env_config.evader_observation_constraints['airspeed_max']
            min_spawn_distance = self.config["min_spawn_distance"]
            max_spawn_distance = self.config["max_spawn_distance"]  
            evader = Evader(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    0.0,
                    0.0,
                    np.random.uniform(self.config["z_bounds"][0], 
                                      self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"]
            )
            agents.append(evader)
            
            #pickle the evader state vector
            counter += 1

        for i in range(self.config["num_pursuers"]):
            min_speed = env_config.pursuer_observation_constraints['airspeed_min']
            max_speed = env_config.pursuer_observation_constraints['airspeed_max']
            min_spawn = self.config["min_spawn_distance"]
            max_spawn = self.config["max_spawn_distance"]
        
            #spawn pursuers at a distance from the evader
            random_spawn_distance = np.random.uniform(min_spawn, max_spawn)
            #random value of -1 or 1
            evader: Evader = agents[0]
            random_angle = evader.state_vector.yaw_rad + np.random.uniform(-np.pi/6, np.pi/6)
            x = evader.state_vector.x + random_spawn_distance * np.cos(random_angle)
            y = evader.state_vector.y + random_spawn_distance * np.sin(random_angle)
            
            #get evader heading
            evader_heading = evader.state_vector.yaw_rad
            # pursuer_heading = evader_heading + np.random.uniform(-np.pi/4, np.pi/4)
            pursuer_heading = np.arctan2(evader.state_vector.y - y, 
                                         evader.state_vector.x - x)
            # pursuer_heading = np.arctan2(y - evader.state_vector.y, 
            #                              x - evader.state_vector.x)
            pursuer = pursuer_heading + np.random.uniform(-np.pi/4, np.pi/4)
            
            pursuer = Pursuer(
                battle_space=self.battlespace,
                state_vector=StateVector(
                    x,
                    y,
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    0.0,
                    0.0,
                    pursuer_heading,
                    np.random.uniform(min_speed, max_speed)
                ),
                id=counter,
                radius_bubble=self.config["bubble_radius"],
                capture_distance=self.config["capture_radius"]
            )
            if self.config["ai_pursuers"]:
                pursuer.is_controlled = True
            agents.append(pursuer)
            counter += 1            
        return agents
        
    def denormalize_action(self, action: np.ndarray, agent_id: int) -> np.ndarray:
        agent: Agent = self.battlespace.agents[agent_id]
        action_space = self.get_agent_action_space(agent_id)
        low = action_space.low
        high = action_space.high
        # Scale the action from [-1, 1] to the original action space range
        action = low + (0.5 * (action + 1.0) * (high - low))
        return np.clip(action, low, high)
    
    def denormalize_observation(self, observation: np.ndarray, agent_id: int) -> np.ndarray:
        agent: Agent = self.battlespace.agents[agent_id]
        obs_space = self.get_agent_observation_space(agent_id)
        low = obs_space.low
        high = obs_space.high
        observation = low + (0.5 * (observation + 1.0) * (high - low))
        return np.clip(observation, low, high)
    
    def normalize_observation(self, observation: np.ndarray, agent_id: int) -> np.ndarray:
        """
        This assumes the observation space for all environments are the same
        """
        #check if self.battlespace is not None
        if self.battlespace is None:
            raise ValueError("Battlespace is None")
        
        agent: Agent = self.battlespace.agents[agent_id]
        observation_space = self.get_agent_observation_space(agent_id)
        low = observation_space.low
        high = observation_space.high
        # Scale the observation to [-1, 1]
        observation = 2.0 * ((observation - low) / (high - low)) - 1.0
        return np.clip(observation, -1.0, 1.0)
    
    def init_action_spaces(self, use_stable_baselines:bool) -> spaces.Dict:
        action_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            if agent.is_controlled:
                
                action_space = self.get_agent_action_space(
                    agent_id=agent.id)                
                if use_stable_baselines:

                    norm_action_space = spaces.Box(
                        low=-1.0, high=1.0, shape=action_space.shape, 
                        dtype=np.float32)
                    return norm_action_space
                else: 
                    return self.get_agent_action_space(agent_id=agent.id)
            
        # return spaces.Dict(action_spaces)

    def map_config(self, action_config:Dict) -> Tuple[List,List]:
        high = []
        low = []
        for k,v in action_config.items():
            if 'max' in k:
                high.append(v)
            elif 'min' in k:
                low.append(v)

        return high, low

    def get_agent_action_space(self, agent_id:int) -> spaces.Box:
        high_action = []
        low_action = []
        agent : Agent = self.battlespace.agents[agent_id]
        if agent.is_pursuer:
            action_config = env_config.pursuer_control_constraints
            high_action, low_action = self.map_config(action_config)
        else:
            action_config = env_config.evader_control_constraints
            high_action, low_action = self.map_config(action_config)
            
        return spaces.Box(low=np.array(low_action), 
                          high=np.array(high_action), 
                          dtype=np.float32)
     
    def init_observation_spaces(self) -> spaces.Dict:
        observation_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            #TODO: Need to refactor since PN will be a policy, that shit's too hard
            if agent.is_controlled:
                observation_spaces[agent.id] = self.get_agent_observation_space(
                    agent_id=agent.id
                )
                return self.get_agent_observation_space(agent_id=agent.id)
                
        #return spaces.Dict(observation_spaces)
    
    def get_agent_observation_space(self, agent_id:int) -> spaces.Box:
        high_obs = []
        low_obs = []
        agent : Agent = self.battlespace.agents[agent_id]
        if agent.is_pursuer:
            obs_config = env_config.pursuer_observation_constraints
            high_obs, low_obs = self.map_config(obs_config)
            
            n_evaders = len(self.agents)
            # we need to store the relative position, heading, and velocity of evader
            # adding this into the observation space
            for i in range(n_evaders):
                low_rel_pos  = env_config.LOW_REL_POS
                high_rel_pos = env_config.HIGH_REL_POS
                low_rel_vel  = env_config.LOW_REL_VEL
                high_rel_vel = env_config.HIGH_REL_VEL
                low_rel_heading = env_config.LOW_REL_ATT
                high_rel_heading = env_config.HIGH_REL_ATT
                low = [low_rel_pos, low_rel_vel, low_rel_heading]
                high = [high_rel_pos, high_rel_vel, high_rel_heading]
                low_obs.extend(low)
                high_obs.extend(high)
        else:
            obs_config = env_config.evader_observation_constraints
            high_obs, low_obs = self.map_config(obs_config)
            n_pursuers = self.config["num_pursuers"]
            for i in range(n_pursuers):
                low_rel_pos  = env_config.LOW_REL_POS
                high_rel_pos = env_config.HIGH_REL_POS
                low_rel_vel  = env_config.LOW_REL_VEL
                high_rel_vel = env_config.HIGH_REL_VEL
                low_rel_heading = env_config.LOW_REL_ATT
                high_rel_heading = env_config.HIGH_REL_ATT
                low = [low_rel_pos, low_rel_vel, low_rel_heading]
                high = [high_rel_pos, high_rel_vel, high_rel_heading]
                low_obs.extend(low)
                high_obs.extend(high)
        
        return spaces.Box(low=np.array(low_obs), 
                          high=np.array(high_obs), 
                          dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        # Set the seed for reproducibility

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        
        self.battlespace = self.__init_battlespace()
        self.all_agents = self.__init_agents()
        self.battlespace.agents = self.all_agents
        self.current_step = 0
        self.agents = [agent for agent in self.all_agents 
                                    if agent.is_controlled and 
                                    not agent.is_pursuer]
        
        self.terminateds = False
        self.truncateds = False
    
        observation = self.get_current_observation(agent_id=0)        
        infos = {}
        #convert to numpy float32
        return observation, infos

    def get_current_observation(self, agent_id:int,
                                get_norm_obs:bool=False) -> np.ndarray:
        """
        """
        agent = self.battlespace.agents[agent_id]
        observation = agent.get_observation()
    
        agent: Evader
        for other_agent in self.all_agents:
            if not other_agent.is_pursuer and other_agent.id == agent_id:
                continue
            
            other_agent: Pursuer
            rel_distance = agent.state_vector.distance_2D(other_agent.state_vector)
            rel_velocity = other_agent.state_vector.speed - agent.state_vector.speed
            rel_heading = agent.state_vector.heading_difference(other_agent.state_vector)
            relative_info = [rel_distance, rel_velocity, rel_heading]
            relative_info[0] = np.clip(relative_info[0], 
                                        env_config.LOW_REL_POS, 
                                        env_config.HIGH_REL_POS)
            relative_info[1] = np.clip(relative_info[1],
                                            env_config.LOW_REL_VEL,
                                            env_config.HIGH_REL_VEL)
            relative_info[2] = np.clip(relative_info[2],
                                            env_config.LOW_REL_ATT,
                                            env_config.HIGH_REL_ATT)
            observation = np.append(observation, relative_info)

        #check if shape is correct
        #assert observation.shape[0] == self.observation_space[agent_id].shape[0]
        if observation.shape[0] != self.observation_space.shape[0]:
            raise ValueError("Observation shape is not correct", observation.shape,
                             self.observation_space.shape[0])

        if get_norm_obs and self.upload_norm_obs:
            observation = normalize_obs(observation, self.vec_env)
            return observation.astype(np.float32)

        #return as type float32
        return observation.astype(np.float32)
    
    def simulate(self, action_dict:np.ndarray, use_multi:bool=True) -> None:
        self.battlespace.act(action_dict, use_multi)
        self.battlespace.step(self.dt)
    
    def step(self, actions:np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.reward = 0
        info = {}
        info['caught'] = False
        self.current_step += 1
        
        if self.use_stable_baselines:
            actions = self.denormalize_action(actions, agent_id=0)
        
        # #do a control based on the control frequency
        if self.current_step % self.control_freq == 0 or self.current_step == 1:
            self.simulate(actions, use_multi=False)
            self.old_action = actions
        else:
            #repeat the last action
            self.simulate(self.old_action, use_multi=False)
            
        self.observation = self.get_current_observation(agent_id=0)        
        controlled_agent: Evader = self.agents[0]
        self.reward = controlled_agent.get_reward(self.observation)
        for agent in self.agents:
            agent: Agent 
            if agent.crashed:
                #terminate the entire episode
                self.terminateds = True
                self.truncateds = True
                self.current_sim_num += 1
                # this is the capture reward
                if agent.is_pursuer == True:
                    self.reward += self.terminal_reward
                else:
                    self.reward += -self.terminal_reward
                    info['caught'] = True
                    
        # check if the episode is done
        if self.current_step >= self.time_steps:
            # print("you win")
            self.terminateds= True
            self.truncateds= True
            self.current_sim_num += 1
            self.num_wins += 1
            self.reward += self.terminal_reward
        
        if self.upload_norm_obs and self.vec_env is not None:
            self.observation = normalize_obs(self.observation, self.vec_env)           
        
        return self.observation, self.reward, self.terminateds, self.truncateds, info
        
    def render(self, mode:str='human') -> None:
        """
        Parameters
        ----------
        mode : str, optional
            The mode of rendering, by default 'human'
        """
        # print(f"Rendering in {mode} mode.")
        pass
        
    def close(self) -> None:
        pass
    
class HRLBattleEnv(AbstractBattleEnv):
    """
    Should have all the abstract methods
    - You set a bunch of boolean parameters on what policies 
    you want to run example:
        - AvoidancePolicy from Pursuers (Evader)
        - EngagePolicy (Evader)
            - Define if you want to use your heuristics or use the RL policy
        - Obstacle Avoidance (Evader)
            - Define if you want to use your heuristics or use the RL policy
    - Observations should be concatenated
        - Requirement is that they should all have the ego observation
            - [Ego, Avoidance, Engage, Obstacle] 
        - Call out each observation space
    """
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None,
                 upload_norm_obs:bool=False,
                 vec_env:VecNormalize=None,
                 evade_env:VecNormalize=None,
                 avoidance_policy:PPO=None,
                 engage_env:VecNormalize=None) -> None:
        super().__init__(spawn_own_space=spawn_own_space,
                         spawn_own_agents=spawn_own_agents,
                         battlespace=battlespace,
                         use_stable_baselines=use_stable_baselines,
                         agents=agents,
                         upload_norm_obs=upload_norm_obs,
                         vec_env=vec_env)
        
        #TODO: Need to refactor this
        model_name = "PPO_evader_2D_280000_steps"
        vec_normalize_path = "PPO_evader_2D_vecnormalize_280000.pkl"

        env = DummyVecEnv([self.create_battle_env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
        # Load the trained model
        self.evader_policy = PPO.load(model_name, 
            env=env, print_system_info=True,
            device='cuda')
        
        self.evade_env = ThreatAvoidEnv(upload_norm_obs=True, vec_env=env)
        self.evade_obs, _ = self.evade_env.reset()

        # if evade_env is not None:
        #     self.evade_env = BattleEnv(upload_norm_obs=True,
        #                                     vec_env=evade_env)
        # else:
        #     raise ValueError("Avoidance environment is None")
            
        # if avoidance_policy is not None:
        #     self.avoidance_policy = avoidance_policy
        # else:
        #     raise ValueError("Avoidance policy is None")
        
        if engage_env is not None:
            self.engage_env = engage_env
        else:
            #use the heuristic policy
            self.engage_env = EngagementEnv(
                spawn_own_agents=True,
                agents = self.evade_env.all_agents,
                use_heuristic_policy=True)
                
        self.update_target()
        # Select between offensive and defensive
        # 0 is defensive, 1 is offensive
        self.action_space = spaces.Discrete(2)  
        self.observation_space = self.init_observation_spaces()
        self.current_env = self.evade_env
        
        self.engage_obs = None
        self.truncateds = False
        self.terminateds = False
        self.control_freq = 3
        
    def create_battle_env(self) -> ThreatAvoidEnv:
        return ThreatAvoidEnv()  # Adjust this to match your environment creation

    def update_target(self) -> None:
        #let's update the target location to spawn
        #wrt to where the pursuer is
        evader:Evader = self.evade_env.agents[0]
        agents = self.evade_env.all_agents
        for agent in agents:
            if agent.is_pursuer:
                pursuer:Pursuer = agent
                break
            
        min_spawn_distance = env_config.MIN_TARGET_DISTANCE
        max_spawn_distance = env_config.MAX_TARGET_DISTANCE
        
        dx = pursuer.state_vector.x - evader.state_vector.x
        dy = pursuer.state_vector.y - evader.state_vector.y
        los = np.arctan2(dy, dx) 
        
        random_distance = np.random.uniform(min_spawn_distance, max_spawn_distance)
        
        random_heading = los + np.random.uniform(-np.pi/4, np.pi/4)
        target_x = pursuer.state_vector.x + (random_distance * np.cos(random_heading))
        target_y = pursuer.state_vector.y + (random_distance * np.sin(random_heading))
        
        #spawn it where the target is closer to the pursuer than the evader
        obstacle = Obstacle(
            x = target_x,
            y = target_y,
            z = 0,
            radius = env_config.TARGET_RADIUS
        )
        
        self.engage_env.make_target(spawn_own_target=True, 
                                    own_target=obstacle)
        
        
    def init_observation_spaces(self) -> spaces.Dict:
        #get the observation space for the avoidance environment
        avoidance_obs_space = self.evade_env.observation_space
        engage_obs_space = self.engage_env.observation_space
        engage_obs_low = engage_obs_space.low[-2:]
        engage_obs_high = engage_obs_space.high[-2:]
        #get the last           
        
        #concatenate the observation spaces
        #obs_space = np.concatenate([avoidance_obs_space, engage_obs_space])
        obs_low = np.concatenate([avoidance_obs_space.low, engage_obs_low])
        obs_high = np.concatenate([avoidance_obs_space.high, engage_obs_high])
        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
    def get_current_observation(self, agent_id:int=0) -> np.ndarray:
        """_summary_
        Method returns the ego observation concatenated with the
        avoidance, engage, and obstacle avoidance observations
        
        Avoidance Policy:
            - The relative position, heading, and velocity of the pursuers
        Engage Policy:
            - The relative position and heading of the target
        
        Returns:
            np.ndarray: _description_
        """
        self.avoid_obs = self.evade_env.get_current_observation(agent_id=0)
        #TODO: need to refactor this gets really confusing because we want to normalize
        self.engage_obs = self.engage_env.get_current_observation(agent_id=0)
        #get the last 2 observations
        # engage_obs = engage_obs[-2:]
        # ego_obs = self.get_current_observation(agent_id=0)        
        obs = np.concatenate([self.avoid_obs, self.engage_obs[-2:]], dtype=np.float32)
        return obs
        
    def reset(self, *, seed=None, options=None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        
        self.evade_env.reset()
        self.engage_env.reset(agents=self.evade_env.all_agents)
        self.update_target()
        self.observation = self.get_current_observation()
        self.current_env = self.evade_env
        self.current_step = 0
        self.reward = 0
        self.truncateds = False
        self.terminateds = False
        
        return self.observation, {}
    
    def get_reward(self) -> float:
        pass
    
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        We will step through both environments
        """
        self.reward = 0
        info = {}
        info['engaged'] = False
        info['caught'] = False
        
        # action = 0
        # #do a control based on the control frequency
        if self.current_step % self.control_freq == 0 or self.current_step == 1:
            current_action = action
            self.old_action = action
        else:
            current_action = self.old_action
        
        if current_action == 0:
            low_level_actions, values = self.evader_policy.predict(
                self.evade_obs, deterministic=True)
            self.evade_obs, avoidance_reward, avoidance_done, _, avoidance_info = \
                self.evade_env.step(low_level_actions)
                
        else:
            if self.engage_env.use_heuristic_policy:
                low_level_actions = self.engage_env.heuristic_policy()
            else:
                low_level_actions, values = self.engage_env.predict(
                    self.engage_obs, deterministic=True)
                
            self.engage_obs, engage_reward, engage_done, _ ,engage_info = \
                self.engage_env.step(low_level_actions)                
            self.evade_obs = self.evade_env.get_current_observation(
                agent_id=0, get_norm_obs=True)
                
        self.observation = self.get_current_observation(agent_id=0)
        ego_agent:Evader = self.evade_env.agents[0]
        
        avoid_reward =  ego_agent.get_reward(
            self.evade_env.get_current_observation(agent_id=0))
        engage_reward = self.engage_env.get_reward(
            self.engage_env.get_current_observation(agent_id=0))
        time_reward = -0.1
        self.reward = avoid_reward + 2.0*engage_reward + time_reward
        
        if self.engage_env.is_close_to_target(ego_agent):
            print("Target reached")
            self.reward = 1000 + self.reward
            self.terminateds = True
            self.truncateds = True
        elif ego_agent.crashed:
            print("Crashed")
            self.reward = -100 + self.reward
            self.terminateds = True
            self.truncateds = True          
        elif self.current_step >= self.time_steps:
            self.reward = -100 + self.reward
            print("Time steps exceeded you lose")
            self.terminateds = True
            self.truncateds = True

        self.current_step += 1
        
        if self.upload_norm_obs and self.vec_env is not None:
            self.observation = normalize_obs(self.observation, 
                                             self.vec_env)
        
        return self.observation, self.reward, self.terminateds, self.truncateds, info
        
    def switch_policy(self, action:int) -> np.ndarray:
        """
        Switch between the avoidance and engage policy
        0 is avoidance, 1 is engage
        """
        norm_avoid_obs = normalize_obs(self.avoid_obs, self.evade_env.vec_env)
        if action == 0:
            print("normalizing avoidance observation", norm_avoid_obs)
            low_level_actions, values = self.avoidance_policy.predict(
                norm_avoid_obs, deterministic=True)
            print("Avoidance policy actions: ", low_level_actions)
        else:
            if self.engage_env.use_heuristic_policy:
                low_level_actions = self.engage_env.heuristic_policy()
            else:
                low_level_actions, values = self.engage_env.predict(
                    self.engage_obs, deterministic=True)
                    
        return low_level_actions