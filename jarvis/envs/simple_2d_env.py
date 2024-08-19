from typing import Dict, Text
from typing import Dict, List, Optional, Text, Tuple, TypeVar
import numpy as np
import gymnasium 
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from jarvis.envs.battle_space import BattleSpace
from jarvis.config import env_config_2d as env_config
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane2D import Agent, Evader, Pursuer

# this is a simple environment that will be used to test the RL algorithms

class BattleEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 use_stable_baselines:bool=True,
                 agents:List[Agent]=None) -> None:
        
        self.config = self.default_config()
        self.dt = env_config.DT
        self.time_steps = env_config.TIME_STEPS
        self.current_step = 0
        self.terminal_reward = 100
        self.use_stable_baselines = use_stable_baselines        
        self.state = None
        self.done = False
        self.control_freq = 2 #(1/control_freq) is the control frequency
        self.old_action = None

        self.current_sim_num = 0
        self.num_wins = 0
        self.mean_episode_length = 0
        
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

        for i in range(self.config["num_pursuers"]):
            min_speed = env_config.pursuer_observation_constraints['airspeed_min']
            max_speed = env_config.pursuer_observation_constraints['airspeed_max']
            min_spawn = self.config["min_spawn_distance"]
            max_spawn = self.config["max_spawn_distance"]
        
            #spawn pursuers at a distance from the evader
            random_spawn_distance = np.random.uniform(min_spawn, max_spawn)
            #random value of -1 or 1
            evader: Evader = agents[0]
            x = evader.state_vector.x + random_spawn_distance * np.random.choice([-1, 1])
            y = evader.state_vector.y + random_spawn_distance * np.random.choice([-1, 1])
            #get evader heading
            evader_heading = evader.state_vector.yaw_rad
            # pursuer_heading = evader_heading + np.random.uniform(-np.pi/4, np.pi/4)
            pursuer_heading = np.arctan2(y - evader.state_vector.y, 
                                         x - evader.state_vector.x)
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
