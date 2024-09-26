import os 
# go back to the root directory
import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.envs.simple_2d_env import RCSEnv
from jarvis.assets.Radar2D import Radar2D, RadarParameters
from jarvis.assets.Radar2D import RadarSystem2D
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane2D import Pursuer, Evader
import seaborn as sns

class TestSingleRadar(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RCSEnv(use_stable_baselines=True)
        self.num_pursuers = 0
        radar_parameters = RadarParameters(false_alarm_rate=0.1,
                                           position=StateVector(0,0,0,0,0,0,0),
                                           max_fov_dg=30,
                                           range_m=500,
                                           c1=-0.25,
                                           c2=1000)        
        self.radar = Radar2D(radar_parameters=radar_parameters,
                             is_circle=True)
        
        radar_parameters_2 = RadarParameters(false_alarm_rate=0.1,
                                           position=StateVector(250,250,0,0,0,0,0),
                                           max_fov_dg=30,
                                           range_m=500,
                                           c1=-0.25,
                                           c2=1000)
        self.radar_2 = Radar2D(radar_parameters=radar_parameters_2,
                                is_circle=True)
        
        self.radar_system = RadarSystem2D([self.radar, self.radar_2])
        
        self.evader = Evader(self.env.battlespace,
                             StateVector(-1000,-1000,0,0,0,0,0),
                             id = 1,
                             radius_bubble=5)
        
    def test_agent_close_from_radar(self) -> None:
        """
        Spawn agent close to radar and check 
        if the radar detects the agent should be high 
        Note to self keep values c2 to be between 500 to 1000
        
        Let's 
         
        """
        distances = np.linspace(0, 1000, 20)
        
        c1_vals = np.arange(-0.25, -0.1, 0.02)
        c2_vals = np.arange(0, 1000, 20)
        fig, ax = plt.subplots()
        
        #set the color as a gradient
        
        for c in c1_vals:
            probabilities = []
            self.radar.c1 = c
            for distance in distances:
                target_position = StateVector(
                    distance, distance, 0, 0, 0, 0, 0)
                rcs_val = self.evader.rcs_table[str(1)]
                probability = self.radar.probability_of_detection(
                    target_position, rcs_val)
                probabilities.append(probability)
            ax.plot(distances, probabilities, label="c1: {}".format(self.radar.c1))
                    
        ax.set_xlabel("Distance (m)")
        ax.legend()
        # plt.show()

    def test_radar_system(self) -> None:
        """
        """
        detections = []
        incident_angles = []
        #set the color as a gradient for distance
        self.evader.state_vector.x = 1000
        self.evader.state_vector.y = 1000
        self.evader.state_vector.z = 0
        
        for i in range(180):
            self.evader.state_vector.yaw_rad = np.deg2rad(i)
            for radar in self.radar_system.radars:
                
                incident_angle = self.radar_system.compute_angle_of_incidence(
                    radar, self.evader)
                # individual_detection = radar.probability_of_detection(
                #     self.evader.state_vector, 0.1)
            
            detection_value = self.radar_system.probability_of_detection_system(
                self.evader)
            print("Detection value: ", detection_value)
            detections.append(detection_value)
            incident_angles.append(incident_angle)
        
        fig, ax = plt.subplots()
        
        # set color gradient pallete for distance
        ax.plot(incident_angles, detections)
        # ax.plot(incident_angles, detections)
        ax.set_xlabel("Yaw angle (deg)")
        ax.set_ylabel("Detection probability")
                    
        # ax.set_xlabel("Distance (m)")
        ax.legend()
        plt.show()
        
        
if __name__ == "__main__":
    unittest.main()
    