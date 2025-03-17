# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from jarvis.envs.simple_agent import PlaneKinematicModel
from typing import List

X_IDX:int = 0
Y_IDX:int = 1
Z_IDX:int = 2
PSI_IDX:int = 5
VEL_IDX:int = 6

def noisy_measurement(x: np.array, pos_noise:float,
                      psi_noise:float, 
                      vel_noise:float,
                      num_measurements:int) -> np.array:
    """
    Generate a noisy measurement from the true state x.
    The measurements are positions, heading and velocity.
    """
    measurements:np.array = np.zeros(num_measurements)
    measurements[0] = x[X_IDX] + np.random.randn() * pos_noise
    measurements[1] = x[Y_IDX] + np.random.randn() * pos_noise
    measurements[2] = x[Z_IDX] + np.random.randn() * pos_noise
    measurements[3] = x[PSI_IDX] + np.random.randn() * psi_noise
    measurements[4] = x[VEL_IDX] + np.random.randn() * vel_noise
    
    return measurements

class UKFPlane():
    def __init__(self,
                 dt:float = 0.1):
        self.n_states:int = 8
        self.dt:float = dt
        self.n_measurements:int = 5
        self.points:MerweScaledSigmaPoints = MerweScaledSigmaPoints(
            n=self.n_states, alpha=.1, beta=2., kappa=1.0)
        
        self.ukf = UKF(dim_x=self.n_states, 
                  dim_z=self.n_measurements, 
                  fx=self.fx,
                hx=self.hx, dt=dt, points=self.points)
        self.ukf.P *= 5  # initial uncertainty
        pos_var:float = 5.0
        psi_var:float = np.deg2rad(3)
        vel_var:float = 0.1
        self.ukf.R = np.diag([pos_var, pos_var, pos_var, psi_var, vel_var])
        # ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.02)
        # remember smaller Q means I trust the model more
        # ukf.Q = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02])
        Q_vals = 1e-5
        self.ukf.Q = np.diag([Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals])



    def set_starting_state(self, x:np.array):
        self.ukf.x = x

    def fx(self, 
           x: np.array, 
           dt: float) -> np.array:
        """
        State transition for an aircraft with an acceleration model, where acceleration is treated as a random walk.
        
        The state vector is:
            [x, y, z, phi, theta, psi, v, a]
        
        Dynamics:
        - Position updated with constant acceleration formula.
        - Attitude remains unchanged (no control inputs).
        - Velocity is updated by integrating acceleration.
        - Acceleration is assumed to be constant in the model, but its process noise is tuned to allow changes.
        """
        # Unpack the state
        x_pos   = x[0]
        y_pos   = x[1]
        z_pos   = x[2]
        phi     = x[3]
        theta   = x[4]
        psi     = x[5]
        v       = x[6]
        a       = x[7]
        
        # Compute displacement using constant acceleration formula
        disp = v * dt + 0.5 * a * dt**2
        
        # Update positions
        x_new = x_pos + disp * np.cos(theta) * np.cos(psi)
        y_new = y_pos + disp * np.cos(theta) * np.sin(psi)
        z_new = z_pos + disp * np.sin(theta)
       
        # Attitude remains constant
        phi_new = phi
        theta_new = theta
        psi_new = psi
        
        # Update velocity
        v_new = v + a * dt
        
        # Acceleration is modeled as a random walk: a_new = a (with increased process noise in Q)
        a_new = a
        
        return np.array([x_new, y_new, z_new, 
                         phi_new, theta_new, psi_new, 
                         v_new, a_new])

    def hx(self, x: np.array) -> np.array:
        """
        For now let's just say the only information we have is 
        the position of the plane, the heading, and the magnitude of the velocity.
        """
        x_reading = x[X_IDX]
        y_reading = x[Y_IDX]
        z_reading = x[Z_IDX]
        psi_reading = x[PSI_IDX]
        vx_reading = x[VEL_IDX]
        return np.array([x_reading, y_reading, z_reading, psi_reading, vx_reading])
    
    def predict(self) -> None:
        self.ukf.predict()
    
    def update(self, z:np.array) -> None:
        self.ukf.update(z)
        
    def get_estimate(self) -> np.array:
        return self.ukf.x
    
    def run(self, z: np.array) -> np.array:
        self.predict()
        self.update(z)
        return self.get_estimate()