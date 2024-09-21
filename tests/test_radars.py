import os 
# go back to the root directory
os.chdir("..")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.envs.simple_2d_env import RCSEnv
from jarvis.assets.Radar2D import Radar2D, RadarParameters
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane2D import Pursuer, Evader

class TestSingleRadar(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RCSEnv(use_stable_baselines=True)
        self.num_pursuers = 0
        radar_parameters = RadarParameters(detection_range=1000,
                                           false_alarm_rate=0.1,
                                           position=StateVector(0,0,0,0,0,0,0),
                                           max_fov_dg=30,
                                           range_m=1000,
                                           c1=0.1,
                                           c2=0.1)        
        self.radar = Radar2D(radar_parameters=radar_parameters,
                             is_circle=True)
        
        self.evader = Evader(self.env.battlespace,
                             StateVector(100,100,0,0,0,0,0))
        
    def test_agent_close_from_radar(self) -> None:
        """
        Spawn agent close to radar and check 
        if the radar detects the agent should be high 
        """
        distances = np.linspace(0, 1000, 100)
        probabilities = []
        
        for distance in distances:
            target_position = StateVector(distance, distance, 0, 0, 0, 0, 0)
            rcs_val = 0.1
            probability = self.radar.probability_of_detection(target_position, rcs_val)
            probabilities.append(probability)
            
        fig, ax = plt.subplots()
        ax.plot(distances, probabilities)
        ax.set_xlabel("Distance (m)")
        
        

    def test_agent_far_from_radar(self) -> None:
        """
        
        """
