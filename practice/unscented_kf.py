"""
I have no fucking clue how a UKF works. Going to try to figure out how to use it
for baseline comparison with the predictformer.

https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

plt.close('all')

def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""
    
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    return F @ x

def h_cv(x):
    return x[[0, 2]]

dt = 0.1
sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
ukf = UKF(dim_x=4, dim_z=2, fx=f_cv,
          hx=h_cv, dt=dt, points=sigmas)
ukf.x = np.array([0., 0., 0., 0.])
ukf.R = np.diag([0.09, 0.09]) 
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

num_steps: int = 100
zs = []
uxs = []
for i in range(num_steps):
    noise_magnitude: float = 1.5
    z = np.array([i + np.random.randn()*noise_magnitude, 
                  i + np.random.randn()*noise_magnitude])
    ukf.predict()
    ukf.update(z)
    uxs.append(ukf.x.copy())
    zs.append(z)
        
uxs = np.array(uxs)
zs = np.array(zs)

fig, ax = plt.subplots(1, 1)
ax.plot(uxs[:, 0], uxs[:, 2], label='UKF')
ax.plot(zs[:, 0], zs[:, 1], label='measurement', marker='x', ls='')
ax.legend()
plt.show()
# print(f'UKF standard deviation {np.std(uxs - xs):.3f} meters')
