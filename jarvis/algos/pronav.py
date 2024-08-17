import numpy as np
from jarvis.utils.Vector import StateVector, PositionVector
from jarvis.config import env_config
def normalize_safe(vector:StateVector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Return the original vector if the magnitude is zero
    return vector / magnitude  # Normalize by dividing each component by the magnitud

import numpy as np

class ProNav():
    def __init__(self, dt: float, nav_constant: float = 4.0) -> None:
        self.dt = dt
        self.nav_constant = nav_constant
        self.Nt = 9.8 * dt  # Assuming env_config.DT is the same as dt
        self.previous_ego_state = None
        self.previous_target_state = None
        self.previous_target_velocity = None  # To store previous target velocity
        self.capture_distance = 10.0  # Distance to target to consider it captured

    def pursuit(self, ego: StateVector, target: StateVector) -> np.ndarray:
        """
        Use pure pursuit to navigate the missile to the target.
        """
        los = target - ego
        # los_unit_vector = los.array[:3] / np.linalg.norm(los.array[:3])
        # ego_speed = ego.speed
        # lateral_acceleration = ego.speed * los_unit_vector 
        heading_cmd = np.arctan2(los.array[1], los.array[0])
        flight_path_rate = self.nav_constant * heading_cmd
        acmd = self.nav_constant * ego.speed
        
        #check dot product and distance
        error_heading = abs(ego.yaw_rad - heading_cmd)
        #wrap between -pi and pi
        # if error_heading > np.pi:
        #     error_heading -= 2*np.pi
        # if error_heading < -np.pi:
        #     error_heading += 2*np.pi
        target_distance = np.linalg.norm(los.array[:3])
        flight_path_rate = self.nav_constant * (heading_cmd - ego.yaw_rad)

        if error_heading > np.deg2rad(20) and target_distance < self.capture_distance + 30:
            acmd = -1.0
        else:
            acmd = 1.0
        
        if error_heading < np.deg2rad(1):
            flight_path_rate = 0.0
            
        """
        Los is the desired heading of the missle, we actually want to send the error
        to the controller to correct the heading of the missile. The error is the difference
        """
        
        return acmd, flight_path_rate

    def navigate(self, ego: StateVector, target: StateVector) -> np.ndarray:
        
        if self.previous_ego_state is not None and self.previous_target_state is not None:
            RTM_old = self.previous_target_state - self.previous_ego_state
            RTM_new = target - ego
    
            theta_old = np.arctan2(RTM_old.array[1], RTM_old.array[0])
            theta_new = np.arctan2(RTM_new.array[1], RTM_new.array[0])
    
            RTM_new = RTM_new.array[:3] / np.linalg.norm(RTM_new.array[:3])
            RTM_old = RTM_old.array[:3] / np.linalg.norm(RTM_old.array[:3])

            if np.linalg.norm(RTM_old) == 0:
                LOS_Delta = np.array([0, 0, 0])
                LOS_Rate = 0.0
            else:
                LOS_Delta = RTM_new - RTM_old
                LOS_Rate = np.linalg.norm(LOS_Delta)

            # Calculate closing velocity
            Vc = -LOS_Rate
            # Calculate lateral acceleration (latax) using APN with At
            #check if los_rate is close to zero
            if abs(LOS_Rate) == 0:
                #use pure pursuit
                Vc, yaw_cmd = self.pursuit(ego, target)
            else:
                los_dot = (theta_new - theta_old) / self.dt
                yaw_cmd = self.nav_constant * los_dot
        else:
            Vc = 0
            latax = 0
            yaw_desired = latax/ ego.speed
            yaw_cmd = yaw_desired 
                
        # Update previous states for the next iteration
        self.previous_ego_state = ego
        self.previous_target_state = target        
        return Vc, yaw_cmd

        