"""
0 = psi
1 = target_x
2 = target_y
3 = missle_x
4 = missle_y
5 = target_vx
6 = target_vy
7 = missle_vx
8 = missle_vx
9 = missle_vy

N = 3.0  # Navigation constant (tune as needed)
beta_init = 0.0  # Initial heading angle

Initial target position: [target_x, target_y] = [1000, 1000]
Initial Missile position: [missle_x, missle_y] = [0, 0]
Initial velocity magnitude for target and missile: 300 m/s

"""

import matplotlib.pyplot as plt
import numpy as np

# State [psi, target_x, target_y, missle_x, missle_y, target_vx, target_vy,
# missle_vx, missle_vy]


class Agent:
    def __init__(self,
                 psi_rad: float,
                 x: float,
                 y: float,
                 vel_magnitude: float):
        self.psi_rad: float = psi_rad
        self.x: float = x
        self.y: float = y
        self.vel_magnitude: float = vel_magnitude
        self.vel_x: float = vel_magnitude * np.cos(psi_rad)
        self.vel_y: float = vel_magnitude * np.sin(psi_rad)


def compute_relative_positions(some_target: Agent,
                               some_missle: Agent) -> np.array:
    rx: float = some_target.x - some_missle.x
    ry: float = some_target.y - some_missle.y

    return np.array([rx, ry])


def compute_relative_velocities(some_target: Agent,
                                some_missle: Agent) -> np.array:
    vx: float = some_target.vel_x - some_missle.vel_x
    vy: float = some_target.vel_y - some_missle.vel_y

    return np.array([vx, vy])


def compute_line_of_sight(relative_pos: np.array) -> float:
    return np.arctan2(relative_pos[1], relative_pos[0])


def compute_line_of_sight_rate(relative_pos: np.array,
                               relative_vel: np.array) -> float:

    return (relative_pos[0] * relative_vel[1] - relative_pos[1] * relative_vel[0]) / np.linalg.norm(relative_pos)**2


def compute_closing_velocity(relative_pos: np.array,
                             relative_vel: np.array) -> float:
    return -np.dot(relative_pos, relative_vel) / np.linalg.norm(relative_pos)


#### START UP ####
missle = Agent(psi_rad=np.deg2rad(0.0),
               x=0.0,
               y=0.0,
               vel_magnitude=30.0)

# Initial conditions
target_heading: float = 0.0
target: Agent = Agent(psi_rad=target_heading,
                      x=200.0,
                      y=150.0,
                      vel_magnitude=20.0)

# relative positions
relative_pos = compute_relative_positions(some_target=target,
                                          some_missle=missle)

# line of sight
lam: float = np.arctan2(relative_pos[1], relative_pos[0])

# missle lead angle
lead_angle = np.arcsin(target.vel_magnitude *
                       np.sin(missle.psi_rad + lam) / missle.vel_magnitude)
vel_missle_x: float = missle.vel_magnitude * \
    np.cos(missle.psi_rad + lead_angle + missle.psi_rad)
vel_missle_y: float = missle.vel_magnitude * \
    np.sin(missle.psi_rad + lead_angle + missle.psi_rad)

states = np.array([missle.psi_rad, relative_pos[0], relative_pos[1], missle.x, missle.y,
                   target.vel_x, target.vel_y, vel_missle_x, vel_missle_y])

dt = 0.05
n_steps = 500


def euler_step(state: np.array, dt: float, N: float) -> np.array:
    """
    Compute the next state using Euler's method.

    Parameters:
      state: The current state of the system.
      dt   : The time step.

    Returns:
      The next state of the system.
    """
    PSI_IDX = 0
    TARGET_X_IDX = 1
    TARGET_Y_IDX = 2
    MISSLE_X_IDX = 3
    MISSLE_Y_IDX = 4
    TARGET_VX_IDX = 5
    TARGET_VY_IDX = 6
    MISSLE_VX_IDX = 7
    MISSLE_VY_IDX = 8
    # Compute the next state using Euler's method
    acc: float = 0.0
    # for now say acceleration is 0 or constant
    vel_mag: float = np.sqrt(state[MISSLE_VX_IDX]**2 + state[MISSLE_VY_IDX]**2)
    psi: float = np.arctan2(state[MISSLE_VY_IDX], state[MISSLE_VX_IDX])

    # relative distance
    rx: float = state[TARGET_X_IDX] - state[MISSLE_X_IDX]
    ry: float = state[TARGET_Y_IDX] - state[MISSLE_Y_IDX]
    rel_pos = np.array([rx, ry])

    rel_vx = state[TARGET_VX_IDX] - state[MISSLE_VX_IDX]
    rel_vy = state[TARGET_VY_IDX] - state[MISSLE_VY_IDX]
    rel_vel = np.array([rel_vx, rel_vy])

    rel_mag: float = np.sqrt(rx**2 + ry**2)

    lam: float = np.arctan2(ry, rx)
    lam_dot = compute_line_of_sight_rate(relative_pos=rel_pos,
                                         relative_vel=rel_vel)

    vc = compute_closing_velocity(relative_pos=rel_pos,
                                  relative_vel=rel_vel)

    # create an array the size of the state
    dx = np.zeros_like(state)
    dx[PSI_IDX] = acc/vel_mag
    dx[TARGET_X_IDX] = -vel_mag * np.sin(state[PSI_IDX])
    dx[TARGET_Y_IDX] = vel_mag * np.cos(state[PSI_IDX])
    dx[MISSLE_X_IDX] = state[MISSLE_VX_IDX]
    dx[MISSLE_Y_IDX] = state[MISSLE_VY_IDX]
    dx[TARGET_VX_IDX] = acc * np.sin(state[PSI_IDX])
    dx[TARGET_VY_IDX] = acc * np.cos(state[PSI_IDX])

    # proportional stuff
    # closing gain
    n_c: float = N * vc * lam_dot
    dx[MISSLE_VX_IDX] = -n_c * np.cos(lam)
    dx[MISSLE_VY_IDX] = n_c * np.sin(lam)

    return state + (dt * dx)


current_state = states
state_history = []
for i in range(n_steps):
    current_state = euler_step(current_state, dt, 1.0)
    print("next state", current_state)
    state_history.append(current_state)

    current_distance = np.linalg.norm(
        current_state[1:3] - current_state[3:5])

    if current_distance < 40:
        print("Hit target")
        break

# convert to an array
state_history = np.array(state_history)

distance = np.linalg.norm(
    state_history[:, 1:3] - state_history[:, 3:5], axis=1)


# plot the trajectory
fig, ax = plt.subplots()


ax.plot(state_history[:, 1], state_history[:, 2], label="Target")
ax.plot(state_history[:, 3], state_history[:, 4], label="Missle")
ax.scatter(state_history[0, 1], state_history[0, 2], c='r', label="Start")
ax.scatter(state_history[-1, 1], state_history[-1, 2], c='g', label="End")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

fig, ax = plt.subplots()
ax.plot(distance)


plt.show()
