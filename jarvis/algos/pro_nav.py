import numpy as np
from jarvis.utils.vector import StateVector


def normalize_safe(vector: StateVector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Return the original vector if the magnitude is zero
    return vector / magnitude  # Normalize by dividing each component by the magnitud


GRAVITY = 9.81


class ProNav():
    def __init__(self, dt: float, nav_constant: float = 4.0,
                 capture_distance: float = 10.0) -> None:
        self.dt: float = dt
        self.nav_constant: float = nav_constant
        self.Nt: float = GRAVITY * dt  # Assuming env_config.DT is the same as dt
        self.previous_ego_state: StateVector = None
        self.previous_target_state: StateVector = None
        self.previous_target_velocity: float = None  # To store previous target velocity
        # Distance to target to consider it captured
        self.capture_distance: float = capture_distance

    def pursuit(self, ego: StateVector, target: StateVector) -> np.ndarray:
        """
        Use pure pursuit to navigate the missile to the target.
        """
        los: StateVector = target - ego
        # los_unit_vector = los.array[:3] / np.linalg.norm(los.array[:3])
        # ego_speed = ego.speed
        # lateral_acceleration = ego.speed * los_unit_vector
        heading_cmd: float = np.arctan2(los.array[1], los.array[0])
        flight_path_rate: float = self.nav_constant * heading_cmd
        acmd = self.nav_constant * ego.speed

        # check dot product and distance
        error_heading: float = abs(ego.yaw_rad - heading_cmd)
        target_distance: float = np.linalg.norm(los.array[:3])
        flight_path_rate: float = self.nav_constant * \
            (heading_cmd - ego.yaw_rad)

        if error_heading > np.deg2rad(20) and target_distance < self.capture_distance + 30:
            acmd = -1.0
        else:
            acmd = 1.0

        if error_heading < np.deg2rad(1):
            flight_path_rate = 0.0

        # wrap the heading command between 0 to 2pi
        flight_path_rate = (flight_path_rate + 2 * np.pi) % (2 * np.pi)

        return acmd, flight_path_rate

    def navigate(self, ego: StateVector, target: StateVector) -> np.ndarray:
        """
        This algorithm is used to navigate the missile to the target.
        Provides a switch between pure pursuit and APN.

        Returns the closing velocity and the yaw command.
        """
        if self.previous_ego_state is not None and self.previous_target_state is not None:
            RTM_old: StateVector = self.previous_target_state - self.previous_ego_state
            RTM_new: StateVector = target - ego

            theta_old: float = np.arctan2(RTM_old.array[1], RTM_old.array[0])
            theta_new: float = np.arctan2(RTM_new.array[1], RTM_new.array[0])

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
            # check if los_rate is close to zero
            if abs(LOS_Rate) == 0:
                # use pure pursuit
                Vc, yaw_cmd = self.pursuit(ego, target)
            else:
                los_dot = (theta_new - theta_old) / self.dt
                yaw_cmd = self.nav_constant * los_dot
        else:
            Vc = 0
            latax = 0
            yaw_desired = latax / ego.speed
            yaw_cmd = yaw_desired

        Vc, yaw_cmd = self.pursuit(ego, target)
        # wrap yaw command between 0 and 2pi
        yaw_cmd = (yaw_cmd + 2 * np.pi) % (2 * np.pi)
        print(f"Vc: {Vc}, yaw_cmd: {np.rad2deg(yaw_cmd)}")
        # Update previous states for the next iteration
        self.previous_ego_state = ego
        self.previous_target_state = target
        return Vc, yaw_cmd
