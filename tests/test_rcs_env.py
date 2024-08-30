"""
Test RCS values are being read 
correctly from the environment
"""

import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
from jarvis.envs.simple_2d_env import RCSEnv

class TestRCSEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RCSEnv(use_stable_baselines=True)
        self.num_pursuers = 0
        self.relative_info_size = 3
        self.state_size = 4
        
    def test_agent_close_from_radar(self) -> None:
        """
        Spawn agent close to radar and check 
        if the radar detects the agent should be high 
        """
        agents = self.env.agents[0]
        #assert that the value is high

    def test_agent_far_from_radar(self) -> None:
        """
        
        """

    