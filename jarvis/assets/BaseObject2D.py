from typing import List, Tuple, TYPE_CHECKING
from matplotlib import pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config_2d
from jarvis.assets.Plane2D import Agent
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
    
    
    