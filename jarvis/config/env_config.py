import numpy as np

### Battle Space Config ###
X_BOUNDS = [-2000, 2000]
Y_BOUNDS = [-2000, 2000]
Z_BOUNDS = [30, 150]

### ENV Config ###
NUM_AGENTS = 1
NUM_PURSUERS = 1
TIME_STEPS = 400
MAX_NUM_STEPS = TIME_STEPS
#if False, the pursuers will be controlled by the AI otherwise default to heuristic
AI_PURSUERS = False
USE_PURSUER_HEURISTICS = False
DT = 0.1
CAPTURE_RADIUS = 10
MIN_SPAWN_DISTANCE = 200
MAX_SPAWN_DISTANCE = 400

# Relative min and max observations
LOW_REL_POS = 0.0
HIGH_REL_POS = 2000.0
LOW_REL_VEL = 0.0
HIGH_REL_VEL = 35.0
LOW_REL_ATT = -np.pi
HIGH_REL_ATT = np.pi

### Agent Config ###
# Pursuer
pursuer_observation_constraints = {
    'x_min': X_BOUNDS[0], 
    'x_max': X_BOUNDS[1],
    'y_min': Y_BOUNDS[0],
    'y_max': Y_BOUNDS[1],
    'z_min': Z_BOUNDS[0],
    'z_max': Z_BOUNDS[1],
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
    'x_min': X_BOUNDS[0], 
    'x_max': X_BOUNDS[1],
    'y_min': Y_BOUNDS[0],
    'y_max': Y_BOUNDS[1],
    'z_min': Z_BOUNDS[0],
    'z_max': Z_BOUNDS[1],
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 25
}
    