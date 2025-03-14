import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from jarvis.envs.simple_agent import PlaneKinematicModel
from typing import List
"""
I have no fucking clue how a UKF works. Going to try to figure out how to use it
for baseline comparison with the predictformer.

https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
"""

plt.close('all')

X_IDX:int = 0
Y_IDX:int = 1
Z_IDX:int = 2
PSI_IDX:int = 5
VEL_IDX:int = 6


def fx(x: np.array, dt: float) -> np.array:
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
    
    return np.array([x_new, y_new, z_new, phi_new, theta_new, psi_new, v_new, a_new])


def hx(x: np.array) -> np.array:
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

def noisy_measurement(x: np.array, pos_noise:float,
                      psi_noise:float, 
                      vel_noise:float,
                      num_measurements:int) -> np.array:
    """
    Generate a noisy measurement from the true state x.
    The measurements are positions, heading and velocity.
    """
    measurements:np.array = np.zeros(num_measurements)
    # x[X_IDX] += np.random.randn() * pos_noise
    # x[Y_IDX] += np.random.randn() * pos_noise
    # x[Z_IDX] += np.random.randn() * pos_noise
    
    # x[PSI_IDX] += np.random.randn() * psi_noise
    # x[VEL_IDX] += np.random.randn() * vel_noise
    measurements[0] = x[X_IDX] + np.random.randn() * pos_noise
    measurements[1] = x[Y_IDX] + np.random.randn() * pos_noise
    measurements[2] = x[Z_IDX] + np.random.randn() * pos_noise
    measurements[3] = x[PSI_IDX] + np.random.randn() * psi_noise
    measurements[4] = x[VEL_IDX] + np.random.randn() * vel_noise
    
    return measurements
    

dt = 0.05
sim_time = 30  # seconds
steps = int(sim_time / dt)

plane: PlaneKinematicModel = PlaneKinematicModel(
    dt_val=dt)
n_states:int = 8
n_controls:int = plane.n_controls
n_measurements:int = 5
points:MerweScaledSigmaPoints = MerweScaledSigmaPoints(
    n=n_states, alpha=.1, beta=2., kappa=1.0)
ukf = UKF(dim_x=n_states, dim_z=n_measurements, fx=fx,
          hx=hx, dt=dt, points=points)

#position variarance
pos_var:float = 5.0
psi_var:float = np.deg2rad(3)
vel_var:float = 0.1
ukf.R = np.diag([pos_var, pos_var, pos_var, psi_var, vel_var])
# ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.02)
# remember smaller Q means I trust the model more
# ukf.Q = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02])
Q_vals = 1e-3
ukf.Q = np.diag([Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals, Q_vals])


# Initialize the state estimate and covariance
plane_start = np.array([0, 0, 0, 0, 0, 0, 12, 0])
ukf.x = plane_start # starting at origin with 10 m/s airspeed, zero angles
ukf.P *= 5  # initial uncertainty

# For simulation, define an initial "true" state
x_true = np.copy(ukf.x)

history_true: List[float] = []
history_est: List[float] = []
history_meas: List[float] = []

# Wind: for now we assume no wind
wind = np.array([0, 0, 0])
plane_start = plane_start[:-1]
random_control:bool = True

for i in range(steps):

    if random_control:
        theta_cmd: float = np.random.uniform(-np.pi/8, np.pi/8)
        psi_cmd: float = np.random.uniform(-np.pi/4, np.pi/4)
        v_cmd: float = np.random.uniform(15, 30)
    else:
        theta_cmd: float = np.deg2rad(0)
        psi_cmd: float = np.deg2rad(25)
        v_cmd: float = 12
        
    control_inputs = np.array([theta_cmd, psi_cmd, v_cmd])
    plane_start = plane.rk45(x=plane_start, u=control_inputs, dt=dt)
    
    # get the measurements and add noise
    # Propagate true state using RK4
    #x_true = fx(x_true, dt, control, plane, wind)
    history_true.append(plane_start)

    # Generate noisy measurement: h(x_true) with additive Gaussian noise
    #z = hx(x_true) + np.random.multivariate_normal(mean=np.zeros(n_measurements), cov=ukf.R)
    z = noisy_measurement(x=plane_start, pos_noise=pos_var, 
                          psi_noise=psi_var, 
                          vel_noise=vel_var, 
                          num_measurements=n_measurements)
    history_meas.append(z)

    # UKF predict step: pass control and additional arguments required by fx
    ukf.predict()
    
    # UKF update step using measurement z
    ukf.update(z)
    
    # Save the state estimate
    history_est.append(ukf.x)
    
# Plot the true and estimated positions (North and East)
history_est = np.array(history_est)
history_true = np.array(history_true)
history_meas = np.array(history_meas)

fig, ax = plt.subplots(1, 1)
ax.plot(history_est[:, 0], history_est[:, 1], label='UKF')
ax.plot(history_true[:, 0], history_true[:, 1], label='True')
ax.plot(history_meas[:, 0], history_meas[:, 1], label='Measurement', marker='x', ls='')
ax.legend()

# plot 3d
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.plot(history_est[:, 0], history_est[:, 1], history_est[:, 2], label='UKF')
ax.plot(history_true[:, 0], history_true[:, 1], history_true[:, 2], label='True')
ax.plot(history_meas[:, 0], history_meas[:, 1], history_meas[:, 2], label='Measurement', marker='x', ls='')

ax.legend()

plt.show()