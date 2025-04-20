import numpy as np
from jarvis.utils.vector import StateVector


def normalize_safe(vector: StateVector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Return the original vector if the magnitude is zero
    return vector / magnitude  # Normalize by dividing each component by the magnitud


def safe_normalize(v: np.array, eps: float = 1e-6):
    return v / np.maximum(np.linalg.norm(v), eps)


GRAVITY = 9.81


class ProNavV2():
    """
    The PN law is implemented as:
         a_cmd = N * closing_velocity * (r x (v x r)) / ||r||^3

    The lateral acceleration a_cmd is converted into pitch and yaw commands:
      - Pitch command: arctan2(-a_cmd_z, current_speed)
      - Yaw correction (local delta): arctan2(a_cmd_y, current_speed)

    The global yaw command is then computed by adding the local yaw correction (delta_yaw)
    to the missile’s current global yaw.

    A nominal velocity command is used, and constraints can be applied to each command.
    """

    def __init__(self,
                 N: float = 1.0,
                 dt: float = 0.5,
                 closing_distance: float = 50.0):
        self.N: float = N
        self.close_distance: float = closing_distance
        self.prev_relative_pos: np.array = None
        self.prev_airspeed: float = None
        self.prev_lambda: float = None
        self.prev_r_norn: float = None
        self.dt: float = dt

    def angle_diff(self, current_angle: float, target_angle: float) -> float:
        """
        Returns the signed difference (target_angle - current_angle)
        wrapped to the range [-pi, pi].
        """
        diff = target_angle - current_angle
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def predict(self, current_pos: np.array,
                relative_pos: np.array,
                current_speed: float,
                current_heading: float,
                relative_vel: float,
                dont_predict: bool = False,
                max_vel:float = 35.0,
                consider_yaw: bool = True,
                dt=0.2) -> np.array:
        """
        Runs pure pursuit but predicts the next state of the target
        Because relative is defined as ego - other = relative
        Relative = ego - other
        """
        evader_pos = current_pos - relative_pos
        evader_vel = current_speed - relative_vel

        if dont_predict:
            relative_evader_pos = relative_pos
        else:
            relative_evader_pos = relative_pos + (relative_vel * dt)

        norm_pos = safe_normalize(relative_evader_pos)
        los = np.arctan2(relative_evader_pos[1], relative_evader_pos[0])
        heading_error = self.angle_diff(current_heading, los)
        # compute the flight path rate
        flight_path_rate = self.N * los

        pitch = np.arctan2(-relative_pos[2], np.linalg.norm(relative_pos))

        yaw_cmd = flight_path_rate
        # wrap angle
        if yaw_cmd > np.pi:
            yaw_cmd = yaw_cmd - 2*np.pi
        elif yaw_cmd < -np.pi:
            yaw_cmd = yaw_cmd + 2*np.pi

        distance = np.linalg.norm(relative_pos)
        Kp: float = 0.01
        los_error = np.arctan2(relative_evader_pos[1],
                               relative_evader_pos[0])

        yaw_error = self.angle_diff(current_heading, los_error)
        
        if consider_yaw:
            if yaw_error > np.deg2rad(45):
                velocity_cmd = 15.0
            else:
                velocity_cmd = max_vel
        else:
            velocity_cmd = max_vel

        return np.array([relative_pos[2], yaw_cmd, velocity_cmd], dtype=np.float32)

    def compute_commands(self, relative_pos: np.array,
                         current_yaw: float,
                         current_speed: float) -> None:
        """
        Args:
            relative_pos (np.array): Relative position vector (3,)
            current_yaw (float): Current yaw angle in radians between -pi and pi
        """

        # check that relative pos is shape (3,)
        print(f"relative pos: {relative_pos}")
        # if relative_pos.shape != (3,):
        #     raise ValueError("Relative position must have shape (3,)")

        r = relative_pos
        rx: float = r[0]
        ry: float = r[1]
        r_norm: float = np.linalg.norm(r)

        # Compute current LOS angle (e.g., horizontal)
        current_lambda = np.arctan2(ry, rx)
        print(f"current lambda: {np.rad2deg(current_lambda)}")
        # Estimate LOS rate and range rate if we have a previous observation.
        if self.prev_relative_pos is not None:
            prev_r = self.prev_relative_pos
            prev_v = self.prev_airspeed
            prev_r_norm = self.prev_r_norn
            prev_lambda = self.prev_lambda

            los_rate = (current_lambda - prev_lambda) / self.dt
            closing_velocity = -(r_norm - prev_r_norm) / self.dt
        else:
            # If no previous observation, initialize estimates to zero.
            los_rate = 0.0
            closing_velocity = current_speed  # assume full airspeed as closing velocity

        self.prev_relative_pos = r
        self.prev_airspeed = current_speed
        self.prev_lambda = current_lambda
        self.prev_r_norn = r_norm
        print("los rate", los_rate)

        if abs(los_rate) < 1.0:
            print("los rate", los_rate)
            los_rate = current_lambda

        a_cmd: float = self.N * closing_velocity * -los_rate
        yaw_correction: float = np.arctan2(a_cmd, current_speed)
        # yaw_cmd: float = current_yaw + yaw_correction
        yaw_cmd: float = yaw_correction
        # wrap yaw command between -pi and pi
        if yaw_cmd > np.pi:
            yaw_cmd = yaw_cmd - 2*np.pi
        elif yaw_cmd < -np.pi:
            yaw_cmd = yaw_cmd + 2*np.pi

        # set the pitch command as a dz component
        pitch_cmd: float = np.arctan2(r[2], r_norm)

        if r_norm > self.close_distance:
            # If we are close to the target, stop the missile
            velocity_cmd: float = 35.0
        else:
            # Otherwise, set a nominal velocity
            velocity_cmd: float = 15.0

        command: np.array = np.array(
            [pitch_cmd, yaw_cmd, velocity_cmd], dtype=np.float32)
        return command

    def calculate(self,
                  relative_pos: np.array,
                  current_yaw: float,
                  current_speed: float,
                  evader_yaw: float,
                  evader_speed: float) -> np.array:
        distance: float = np.linalg.norm(relative_pos[0:1])

        dx: float = relative_pos[0]
        dy: float = relative_pos[1]

        evader_vx: float = evader_speed * np.cos(evader_yaw)
        evader_vy: float = evader_speed * np.sin(evader_yaw)
        pursuer_vx: float = current_speed * np.cos(current_yaw)
        pursuer_vy: float = current_speed * np.sin(current_yaw)

        delta_vx: float = evader_vx - pursuer_vx
        delta_vy: float = evader_vy - pursuer_vy

        r_dot: float = ((dx * delta_vx) + (dy * delta_vy)) / distance
        Vc: float = -r_dot

        cos_psi = np.cos(current_yaw)
        sin_psi = np.sin(current_yaw)

        Cwp = np.array([[cos_psi, sin_psi], [-sin_psi, cos_psi]])

        rel_vel_global = np.array([delta_vx, delta_vy])

        rel_pos_body = np.dot(Cwp, np.array([dx, dy]))
        rel_vel_body = np.dot(Cwp, rel_vel_global)

        x_body = rel_pos_body[0]
        y_body = rel_pos_body[1]
        lam = np.arctan2(y_body, x_body)

        x_dot_body = rel_vel_body[0]
        y_dot_body = rel_vel_body[1]

        r_squared = x_body**2 + y_body**2

        lam_dot = (x_body * y_dot_body - y_body * x_dot_body) / \
            np.square(x_body) / np.square(np.cos(lam))

        nL = self.N * Vc * lam_dot

        yaw_rate = nL / current_speed
        delta_yaw = yaw_rate * self.dt
        yaw_cmd = current_yaw + delta_yaw

        # wrap yaw command between -pi and pi
        if yaw_cmd > np.pi:
            yaw_cmd = yaw_cmd - 2*np.pi
        elif yaw_cmd < -np.pi:
            yaw_cmd = yaw_cmd + 2*np.pi

        return np.array([0, yaw_cmd, Vc], dtype=np.float32)

    def augmented_pro_nav(self, relative_pos: np.array,
                          current_yaw: float,
                          current_speed: float) -> np.array:
        """
        This function implements the augmented pro nav algorithm
        """

        norm_pos: np.array = normalize_safe(relative_pos)

        if self.prev_relative_pos is not None:

            Nt: float = GRAVITY * self.dt

            print(f"norm pos: {norm_pos}")
            if np.linalg.norm(norm_pos) == 0:
                # If the relative position is zero, return a zero command
                los_delta = np.array([0, 0, 0])
                los_rate = 0.0
            else:
                los_delta = norm_pos - self.prev_relative_pos
                los_rate = np.linalg.norm(los_delta)

            print("los rate", los_rate)
            # Calculate closing velocity
            Vc = -los_rate

            if los_rate < 0.1:
                los_rate = 0.1

            latax = norm_pos * (self.N * Vc * los_rate) + \
                los_delta * (Nt * 0.5 * self.N)

            print(f"latax: {latax}")
            # Calculate the pitch command
            pitch_cmd: float = np.arctan2(-latax[2], current_speed)

            yaw_cmd = np.arctan2(latax[1], latax[0])
            # wrap yaw command between - pi and pi
            if yaw_cmd > np.pi:
                yaw_cmd = yaw_cmd - 2*np.pi
            elif yaw_cmd < -np.pi:
                yaw_cmd = yaw_cmd + 2*np.pi

            self.prev_relative_pos = norm_pos

            return np.array([pitch_cmd, -yaw_cmd, Vc], dtype=np.float32)

        self.prev_relative_pos = norm_pos
        return np.array([0, 0, current_speed], dtype=np.float32)


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
        # I HAVE TO flip the sign of the lateral acceleration
        heading_cmd: float = np.arctan2(los.array[0], los.array[1])
        flight_path_rate: float = self.nav_constant * heading_cmd
        acmd = self.nav_constant * ego.speed

        error = heading_cmd - ego.yaw_rad
        if error > np.pi:
            error = error - 2*np.pi
        elif error < -np.pi:
            error = error + 2*np.pi

        return acmd, heading_cmd

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

        # Vc, yaw_cmd = self.pursuit(ego, target)
        # wrap yaw command between 0 and 2pi
        # print(f"Vc: {Vc}, yaw_cmd: {np.rad2deg(yaw_cmd)}")
        # Update previous states for the next iteration
        self.previous_ego_state = ego
        self.previous_target_state = target
        return Vc, yaw_cmd
