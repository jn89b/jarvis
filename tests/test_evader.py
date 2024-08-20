import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
import copy
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config
from jarvis.assets.Plane2D import Evader, Pursuer
from jarvis.visualizer.visualizer import Visualizer
from tests.utils import setup_battlespace

class TestEvader(unittest.TestCase):
    """
    Test to see if my 2D evader is working
    """
    def setUp(self):
        self.battle_space = setup_battlespace()
        
        evader_state = StateVector(x=0, y=0, z=0,
            roll_rad=0, pitch_rad=0, yaw_rad=0, speed=20)
        
        self.evader = Evader(
            battle_space=self.battle_space,
            state_vector=evader_state,
            id=0,
            radius_bubble=5,
            is_controlled=True
        )
        
    def test_movement(self)->None:
        # Number of simulation steps
        steps = 100
        dt = env_config.DT
        start_position = copy.deepcopy(self.evader.state_vector)    
        for step in range(steps):
            yaw_cmd = np.deg2rad(10)
            vel_cmd = 20
            act_array = np.array([yaw_cmd, vel_cmd])
            self.evader.act(act_array)
            self.evader.step(dt)
            # print(self.evader.state_vector)
            
        end_position = self.evader.state_vector
        
        #check if numpy array is not equal
        if not np.array_equal(start_position.array, end_position.array):
            self.assertTrue(True)
            
if __name__ == "__main__":
    unittest.main()
