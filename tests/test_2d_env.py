import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.simple_2d_env import BattleEnv
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config
from jarvis.assets.Plane2D import Pursuer, Evader
from tests.utils import setup_battlespace
from jarvis.visualizer.visualizer import Visualizer

"""
Unit test to test the environment with
"""

class Test2DEnv(unittest.TestCase):
    
    def setUp(self) -> None:
        self.env = BattleEnv(use_stable_baselines=True)
        self.num_pursuers = 0
        self.relative_info_size = 3
        self.state_size = 4 # x, y,psi, speed
        self.corect_action_size = 2
        for agent in self.env.all_agents:
            if agent.is_pursuer:
                self.num_pursuers += 1
        self.correct_obs_size = self.state_size + (self.num_pursuers \
            * self.relative_info_size)        
        
        
    def test_size_observation_space(self) -> None:
        #get the agents
        agents = self.env.agents[0]

        #check if the observation space is correct
        self.assertEqual(self.env.observation_space.shape[0], self.correct_obs_size)
        
    def test_size_action_space(self) -> None:
        #get the agents
        agents = self.env.agents[0]

        #check if the action space is correct
        self.assertEqual(self.env.action_space.shape[0], self.corect_action_size)        
        
    def test_environment_correct(self) -> None:
        
    
if __name__ == "__main__":
    unittest.main()
