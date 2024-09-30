import numpy as np

### Battle Space Config ###
X_BOUNDS = [-2000, 2000]
Y_BOUNDS = [-2000, 2000]
Z_BOUNDS = [30, 150]

### Engagement Environment Configuration ###
TARGET_RADIUS = 5
MIN_TARGET_DISTANCE = 150
MAX_TARGET_DISTANCE = 350
TARGET_TIME_STEPS = 850
TARGET_X = 250
TARGET_Y = 250

### Radar Config ###
# MIN_NUM_RADARS = 2
NUM_RADARS_MIN = 3
NUM_RADARS_MAX = 3
RADAR_RANGE = 350
RADAR_FOV = 120
RADAR_SPAWN_MIN_DISTANCE = RADAR_RANGE 
RADAR_SPAWN_MAX_DISTANCE = RADAR_RANGE + RADAR_RANGE/2
RADAR_CAPTURE_DISTANCE = 150 # used 

### Evader  Config ###
NUM_AGENTS = 1
NUM_PURSUERS = 2
TIME_STEPS = 450
MAX_NUM_STEPS = TIME_STEPS
#if False, the pursuers will be controlled by the AI otherwise default to heuristic
AI_PURSUERS = False
USE_PURSUER_HEURISTICS = False
DT = 0.1
CAPTURE_RADIUS = 10
MIN_SPAWN_DISTANCE = 200
MAX_SPAWN_DISTANCE = 350 #300
EFFECTOR_RANGE = 5

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
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 30
}

pursuer_control_constraints = {
    'u_psi_min':  -np.deg2rad(40),
    'u_psi_max':   np.deg2rad(40),
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

# Evader
evader_control_constraints = {
    'u_psi_min':  -np.deg2rad(40),
    'u_psi_max':   np.deg2rad(40),
    'v_cmd_min':   15,
    'v_cmd_max':   25
}

evader_observation_constraints = {
    'x_min': X_BOUNDS[0], 
    'x_max': X_BOUNDS[1],
    'y_min': Y_BOUNDS[0],
    'y_max': Y_BOUNDS[1],
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 25
}