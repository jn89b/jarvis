from typing import List, Tuple, TYPE_CHECKING
from matplotlib import pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config_2d
from Plane2D import Agent
import numpy as np

class BaseObject():
    is_pursuer = False
    def __init__(self,
                 state_vector:StateVector) -> None:
        self.state_vector = state_vector
        
    def __repr__(self) -> str:
        return f"BaseObject({self.state_vector})"
    
    def distance_to(self, other:"Agent", 
                    use_2d:bool=False) -> float:
        if use_2d:
            return self.state_vector.distance_2D(other.state_vector)
        else:
            return self.state_vector.distance_3D(other.state_vector)
    
class Radar(BaseObject):
    def __init__(self,
                 state_vector:StateVector,
                 range:float,
                 fov:float) -> None:
        super().__init__(state_vector)
        self.range = range
        self.fov = fov
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
            
    def get_lower_bound(self) -> Tuple[float, float]:
        x = self.state_vector.x + self.range*np.cos(self.state_vector.yaw_rad - self.fov/2)
        y = self.state_vector.y + self.range*np.sin(self.state_vector.yaw_rad - self.fov/2)
        return np.array([x, y])

    def get_upper_bound(self) -> Tuple[float, float]:
        x = self.state_vector.x + self.range*np.cos(self.state_vector.yaw_rad + self.fov/2)
        y = self.state_vector.y + self.range*np.sin(self.state_vector.yaw_rad + self.fov/2)
        return np.array([x, y])
    
    def detect_probability(self, agent:Agent) -> float:
        distance = self.distance_to(agent)
        if distance < self.range:
            return 1.0
        else:
            return 0.0
        
    def is_in_fov(self, agent:Agent) -> bool:
        distance = self.distance_to(agent)
        #check if within the heading
        heading_diff = np.abs(self.state_vector.yaw_rad - agent.state_vector.yaw_rad)
        if heading_diff < self.fov/2 and distance < self.range:
            return True
        return False
    
    def __repr__(self) -> str:
        return f"Radar({self.state_vector, self.range, self.fov})"
    
class RadarSystem():
    def __init__(self,
                 battle_space:BattleSpace,
                 radars:List[Radar]) -> None:
        self.battle_space = battle_space
        self.radars = radars

    def compute_detection_probability(self, agent:Agent) -> float:
        detection_probability = 0.0
        for radar in self.radars:
            detection_probability += radar.detect_probability(agent)
        return detection_probability
    
    def __repr__(self) -> str:
        return f"RadarSystem({self.radars})"
    
    