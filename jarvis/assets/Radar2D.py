"""
"""
import numpy as np
from typing import List
from dataclasses import dataclass, field
from jarvis.utils.Vector import StateVector
from jarvis.assets.Plane2D import Evader

@dataclass
class RadarParameters:
    """
    Radar parameters.
    """
    def __init__(self,
                 false_alarm_rate: float, 
                 position: StateVector,
                 max_fov_dg:float,
                 range_m:float, 
                 c1:float,
                 c2:float):
        self.false_alarm_rate = false_alarm_rate
        self.position = position
        self.max_fov_dg = max_fov_dg
        self.range_m = range_m
        self.max_fov_rad = np.deg2rad(max_fov_dg)
        self.c1 = c1
        self.c2 = c2
        self.radar_fq_hz = 2000

class Radar2D():
    def __init__(self, radar_parameters: RadarParameters,
                 is_circle:bool = True):
        self.radar_parameters:RadarParameters = radar_parameters
        self.is_circle:bool = is_circle
        self.get_fov()
        self.c1:float = radar_parameters.c1
        self.c2:float = radar_parameters.c2
        self.radar_fq_hz:float = radar_parameters.radar_fq_hz
        
    def get_fov(self) -> None:
        """
        Compute the lateral maximum field of view.
        """
        position = self.radar_parameters.position
        upper_x = position.x + self.radar_parameters.range_m * np.cos(position.yaw_rad + self.radar_parameters.max_fov_rad/2)
        upper_y = position.y + self.radar_parameters.range_m * np.sin(position.yaw_rad + self.radar_parameters.max_fov_rad/2)
        lower_x = position.x + self.radar_parameters.range_m * np.cos(position.yaw_rad - self.radar_parameters.max_fov_rad/2)
        lower_y = position.y + self.radar_parameters.range_m * np.sin(position.yaw_rad - self.radar_parameters.max_fov_rad/2)
        
        self.upper_fov_pos = StateVector(upper_x, upper_y, position.z, 
                                         position.roll_rad, position.pitch_rad, 
                                         position.yaw_rad, position.speed)
        self.lower_fov_pos = StateVector(lower_x, lower_y, position.z, 
                                         position.roll_rad, position.pitch_rad, 
                                         position.yaw_rad, position.speed)
        
        
    def probability_of_detection(self, target_position: StateVector, 
                                 rcs_val:float) -> float:
        """
        Compute the probability of detection.
        If is_circle then use the circle model for the radar.
        """
        distance = target_position.distance_2D(self.radar_parameters.position)
        # if distance >= self.radar_parameters.detection_range:
        #     return 0
        # Compute the probability of detection
        linear_db = 10**(rcs_val/10) 
        radar_prob_detection = 1/(1 +(self.c2* np.power(distance,4) / linear_db)**self.c1)
        probability_detection = 1- pow(radar_prob_detection , self.radar_fq_hz)

        return probability_detection
        
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a 2D vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

class RadarSystem2D():
    def __init__(self, radar_system: List[Radar2D]):
        self.radars = radar_system
        
    def compute_angle_of_incidence(self, radar: Radar2D, agent: Evader) -> float:
        """
        Calculate the angle of incidence between the radar's line of sight and the agent's heading.
        where East is 0 degrees and angles increase counterclockwise.
        """
        
        dx = agent.state_vector.x - radar.radar_parameters.position.x
        dy = agent.state_vector.y - radar.radar_parameters.position.y
        los_angle = np.arctan2(dy, dx)
        
        los_norm = normalize_vector(np.array([dx, dy]))
        
        heading_vector = np.array([np.cos(agent.state_vector.yaw_rad),
                                      np.sin(agent.state_vector.yaw_rad)])
        
        dot_product = np.dot(los_norm, heading_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        #incident angle
        angle_of_incidence_rad = np.arccos(dot_product)
        angle_of_incidence_deg = np.degrees(angle_of_incidence_rad)
        
        return angle_of_incidence_deg
        
        
    def probability_of_detection_system(self, agent: Evader) -> float:
        """
        Compute the probability of detection for a system of radars.
        Each radar is assumed to operate independently.
        """
        non_detection_probs = []
        agent_rcs:dict = agent.rcs_table
        # Loop through each radar and calculate its probability of detection
        for radar in self.radars:
            radar:Radar2D
            angle_incidence_dg = self.compute_angle_of_incidence(radar, agent)
            #round to the nearest integer
            angle_incidence_dg = int(np.round(angle_incidence_dg))
            if str(angle_incidence_dg) in agent_rcs:
                rcs_val = agent_rcs[str(angle_incidence_dg)]
            else:
                raise ValueError("RCS value not found for this incident angle", angle_incidence_dg)
            
            prob_detection = radar.probability_of_detection(agent.state_vector, rcs_val)
            print("Probability of detection:", prob_detection)
            # Probability of not being detected by this radar
            non_detection_probs.append(1 - prob_detection)
        
        # Overall probability of detection
        overall_prob_detection = 1 - np.prod(non_detection_probs)
        
        return overall_prob_detection

