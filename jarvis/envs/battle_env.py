from typing import Dict, Text
from typing import Dict, List, Optional, Text, Tuple, TypeVar
import numpy as np
import gymnasium 
from gymnasium import spaces
from jarvis.envs.battle_space import BattleSpace
from jarvis.config import env_config
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane import Agent

"""
REMEMBER KEEP THIS SIMPLE THEN APPLY MORE COMPLEXITY
#TODO: Add 
- [ ] Terrain
"""

class BattleEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 battlespace:BattleSpace=None,
                 agents:List[Agent]=None) -> None:
        self.config = self.default_config()
        self.dt = env_config.DT
        
        self.state = None
        self.done = False
        self.reward = 0
        self.info = {}
        
        if spawn_own_space and battlespace is None:
            self.battlespace = self.__init_battlespace()
        else:
            self.battlespace = battlespace
        
        if spawn_own_agents and agents is None:
            self.agents = self.__init_agents()
        else:
            self.agents = agents
            # we awnt to control the evaders
            self.controlled_vehicles = [agent for agent in self.agents 
                                        if agent.is_controlled and 
                                        not agent.is_pursuer]
            self.battlespace.agents = self.agents
        
        self.action_spaces = self.init_action_spaces()
        self.observation_spaces = self.init_observation_spaces()
        
    @property
    def vehicle(self) -> Agent:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] \
            if self.controlled_vehicles else None
        
    @classmethod
    def default_config(cls) -> Dict:
        config = {}
        
        config.update({
            "x_bounds": env_config.X_BOUNDS,
            "y_bounds": env_config.Y_BOUNDS,
            "z_bounds": env_config.Z_BOUNDS,
            "num_evaders": env_config.NUM_AGENTS,
            "num_pursuers": env_config.NUM_PURSUERS,
            "use_pursuer_heuristics": env_config.USE_PURSUER_HEURISTICS
        })
        
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
            agents.append(Agent(
                state_vector=StateVector(
                    np.random.uniform(self.config["x_bounds"][0], self.config["x_bounds"][1]),
                    np.random.uniform(self.config["y_bounds"][0], self.config["y_bounds"][1]),
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(0, 100)
                ),
                id=counter,
                radius_bubble=10
            ))
            counter += 1
        
        for i in range(self.config["num_pursuers"]):
            agents.append(Agent(
                state_vector=StateVector(
                    np.random.uniform(self.config["x_bounds"][0], self.config["x_bounds"][1]),
                    np.random.uniform(self.config["y_bounds"][0], self.config["y_bounds"][1]),
                    np.random.uniform(self.config["z_bounds"][0], self.config["z_bounds"][1]),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(0, 100)
                ),
                id=counter,
                radius_bubble=10
            ))
            counter += 1
        
        return agents
        
    def init_action_spaces(self) -> spaces.Dict:
        action_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            # if agent.is_controlled:
            action_spaces[agent.id] = self.get_agent_action_space()
        
        return spaces.Dict(action_spaces)

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
        elif agent.is_evader:
            action_config = env_config.evader_control_constraints
            high_action, low_action = self.map_config(action_config)
            
        return spaces.Box(low=np.array(low_action), 
                          high=np.array(high_action), 
                          dtype=np.float32)
     
    def init_observation_spaces(self) -> spaces.Dict:
        observation_spaces = {}
        for agent in self.battlespace.agents:
            agent: Agent = agent
            #TODO: Need to refactor since PN will be a policy
            if agent.is_controlled:
                observation_spaces[agent.id] = self.get_agent_observation_space()
        return spaces.Dict(observation_spaces)
    
    def get_agent_observation_space(self, agent_id:int) -> spaces.Box:
        high_obs = []
        low_obs = []
        agent : Agent = self.battlespace.agents[agent_id]
        if agent.is_pursuer:
            obs_config = env_config.pursuer_observation_constraints
            high_obs, low_obs = self.map_config(obs_config)
        elif agent.is_evader:
            obs_config = env_config.evader_observation_constraints
            high_obs, low_obs = self.map_config(obs_config)
            
        return spaces.Box(low=np.array(low_obs), 
                          high=np.array(high_obs), 
                          dtype=np.float32)
        
    def simulate(self, action_dict:spaces.Dict) -> None:
        for agent in self.battlespace.agents:
            agent: Agent = agent
            if agent.is_controlled and agent.id in action_dict:
                agent.act(action_dict[agent.id])
                
        # self.battlespace.act(action_dict)
        self.battlespace.step(self.dt)
        
    def step(self, actions:spaces.Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Parameters
        ----------
        action : Dict
            The action taken by the agent.
        
        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict]
            The state, reward, done, and info of the environment.
        """
        self.simulate(actions)

        self.state = np.random.rand(1, 6)
        self.reward = np.random.rand(1)
        self.done = np.random.choice([True, False])
        self.info = {}
        
        return self.state, self.reward, self.done, self.info
    
    def reset(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The initial state of the environment.
        """
        self.state = np.random.rand(1, 6)
        return self.state
    
    def render(self, mode:str='human') -> None:
        """
        Parameters
        ----------
        mode : str, optional
            The mode of rendering, by default 'human'
        """
        print(f"Rendering in {mode} mode.")
        
    def close(self) -> None:
        pass