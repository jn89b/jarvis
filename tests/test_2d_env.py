import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.simple_2d_env import ThreatAvoidEnv
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config
from jarvis.assets.Plane2D import Pursuer, Evader
from tests.utils import setup_battlespace
from jarvis.visualizer.visualizer import Visualizer
from stable_baselines3.common.env_checker import check_env

"""
Unit test to test the environment with
"""

class Test2DEnv(unittest.TestCase):
    
    def setUp(self) -> None:
        self.env = ThreatAvoidEnv(use_stable_baselines=True)
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
        
        #check the bounds of the observation space
        x_low = self.env.observation_space.low[0]
        y_low = self.env.observation_space.low[1]
        psi_low = self.env.observation_space.low[2]
        v_low = self.env.observation_space.low[3]
        
        x_high = self.env.observation_space.high[0]
        y_high = self.env.observation_space.high[1]
        psi_high = self.env.observation_space.high[2]
        v_high = self.env.observation_space.high[3]
        
        print("x_low: ", x_low)
        print("y_low: ", y_low)
        print("psi_low: ", psi_low)
        print("v_low: ", v_low)
        
        print("x_high: ", x_high)
        print("y_high: ", y_high)
        print("psi_high: ", psi_high)
        print("v_high: ", v_high)
        
    def test_size_action_space(self) -> None:
        #get the agents
        agents = self.env.agents[0]

        #check if the action space is correct
        self.assertEqual(self.env.action_space.shape[0], 
                         self.corect_action_size)        
        
    def test_environment_correct(self) -> None:
        check_env(self.env)        
    
    def test_environment_step(self) -> None:
        # Number of simulation steps
        steps = 400
        dt = env_config.DT
        for step in range(steps):
            # action = np.array([np.deg2rad(10), 20])
            print("Step: ", step)
            action = self.env.action_space.sample()
            obs, reward, done, _, info = self.env.step(action)
            
            #self.env.render()
            if done:
                break
            print("\n")        
        #visualize the environment
        battlespace = self.env.battlespace
        data_vis = Visualizer()
        fig, ax = data_vis.plot_2d_trajectory(battlespace)
        fig , ax = data_vis.plot_attitudes2d(battlespace)
        plt.show()
    
    def test_environment_reset(self) -> None:
        self.env.reset()
        self.env.render()
    
if __name__ == "__main__":
    unittest.main()
