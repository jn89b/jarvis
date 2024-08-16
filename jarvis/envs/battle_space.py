"""
Include radars, agents, terrain, etc.
"""
import numpy as np

from typing import List, Tuple, TYPE_CHECKING,Dict
from jarvis.utils.Vector import StateVector
if TYPE_CHECKING:
    from jarvis.assets.Plane import Agent

class BattleSpace():
    def __init__(self,
                 x_bounds:np.ndarray,
                 y_bounds:np.ndarray,
                 z_bounds:np.ndarray,
                 agents:List["Agent"]) -> None:
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.agents = agents
        
    def is_out_bounds(self, state_vector:StateVector) -> bool:
        """
        Check if the state vector is out of bounds.
        """
        x, y, z = state_vector.x, state_vector.y, state_vector.z
        return (x < self.x_bounds[0] or x > self.x_bounds[1] or
                y < self.y_bounds[0] or y > self.y_bounds[1] or
                z < self.z_bounds[0] or z > self.z_bounds[1])
        
    def act(self, action:Dict) -> None:
        """
        Act on the environment.
        """
        for agent in self.battlespace.agents:
            agent.act(action)
            
    def step(self, dt:float) -> None:
        """
        Step the environment.
        """
        for agent in self.agents:
            agent.step(dt)
        
    
        
    
    