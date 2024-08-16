import numpy as np

### Battle Space Config ###
X_BOUNDS = [-1000, 1000]
Y_BOUNDS = [-1000, 1000]
Z_BOUNDS = [0, 150]


### ENV Config ###
NUM_AGENTS = 1
NUM_PURSUERS = 2
USE_PURSUER_HEURISTICS = False
DT = 0.1

### Agent Config ###
# Pursuer
pursuer_observation_constraints = {
    'x_min': -750, 
    'x_max': 750,
    'y_min': -750,
    'y_max': 750,
    'z_min': 30,
    'z_max': 100,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 30
}

pursuer_control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(5),
    'u_theta_max': np.deg2rad(5),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

# Evader
evader_control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(5),
    'u_theta_max': np.deg2rad(5),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   25
}

evader_observation_constraints = {
    'x_min': -750,
    'x_max': 750,
    'y_min': -750,
    'y_max': 750,
    'z_min': 30,
    'z_max': 100,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 25
}
    