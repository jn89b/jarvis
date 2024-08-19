import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config
from jarvis.assets.Plane2D import Pursuer, Evader
from tests.utils import setup_battlespace
from jarvis.visualizer.visualizer import Visualizer
        

class TestProNav2D(unittest.TestCase):
    def setUp(self):
        self.battlespace = setup_battlespace()
        # Initialize Evader
        evader_state = StateVector(x=0, y=0, z=0, 
                                   roll_rad=0, pitch_rad=0, 
                                   yaw_rad=0, speed=20)
        self.evader = Evader(
            battle_space=self.battlespace,
            state_vector=evader_state,
            id=0,
            radius_bubble=5,
            is_controlled=True
        )

        # Initialize Pursuer
        pursuer_state = StateVector(x=-150, y=50, z=0, 
                            roll_rad=0, pitch_rad=0, 
                            yaw_rad=0, speed=20)
        self.pursuer = Pursuer(
            battle_space=self.battlespace,
            state_vector=pursuer_state,
            id=1,
            radius_bubble=5,
            is_controlled=True,
            capture_distance=10
        )
        
        self.battlespace.agents = [self.evader, self.pursuer]

    def test_pro_nav_guidance(self):
        # Number of simulation steps
        steps = 100
        dt = env_config.DT
        initial_distance = self.pursuer.distance_to(self.evader, use_2d=True)
        evader_action = np.array([np.deg2rad(10), 
                                  20])
        self.evader.act(evader_action)
        for step in range(steps):
            # Step the pursuer in the environment
            self.pursuer.step(dt)
            self.evader.step(dt)
            # Check the distance
            new_distance = self.pursuer.distance_to(self.evader, use_2d=True)
            # print(f"Step {step}: Distance = {new_distance}")
            
            # Assert that the pursuer is getting closer to the evader
            self.assertLess(new_distance, initial_distance)
            initial_distance = new_distance
        
        # Check if the pursuer eventually captures the evader
        # self.assertTrue(self.pursuer.is_colliding(self.pursuer.capture_distance))
        #visualize the simulation
        # vis = Visualizer()
        # vis.plot_2d_trajectory(self.battlespace)
        # vis.plot_attitudes2d(self.battlespace)
        # plt.show()
        

if __name__ == "__main__":
    unittest.main()
