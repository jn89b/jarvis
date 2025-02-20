import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from jarvis.envs.simple_agent import PlaneKinematicModel
# Instantiate the PlaneKinematicModel with dt=0.1 and default tau values.

plt.close()
# Define the initial state vector:
# Order: [x, y, z, phi, theta, psi, v, p, q, r]
# Start all at zero.


# psi_cmds = np.arange(np.deg2rad(-180), np.deg2rad(180), np.deg2rad(10))

psi_cmds = np.arange(np.deg2rad(-180), np.deg2rad(180), np.deg2rad(10))
# Define a constant (step) control input:
# Order: [u_phi, u_theta, u_psi, v_cmd]
# Here, we set the attitude commands to 0 and command a step in airspeed to 10 m/s.
roll_cmd: float = np.deg2rad(0)
pitch_cmd: float = np.deg2rad(0)
# Commands are Clockwise positive yaw_cmd will make the plane go right
yaw_cmd: float = np.deg2rad(45)

# Define wind as zero (i.e. [wind_x, wind_y, wind_z])
wind = np.array([0.0, 0.0, 0.0])
history_overall = []
for cmd in psi_cmds:
    x0 = np.zeros(7)
    x0[1] = 20
    x0[2] = 50
    x0[-1] = 15
    # Set simulation parameters
    num_steps = 600

    u = np.array([roll_cmd, pitch_cmd, cmd, 20.0])
    # Initialize storage for simulation data
    state_history = [x0]
    time_history = [0.0]

    # Set current state and time
    x_current = x0.copy()
    t_current = 0.0

    model = PlaneKinematicModel(dt_val=0.05)
    dt = model.dt_val

    # Simulation loop: perform 50 integration steps using RK45
    for i in range(num_steps):
        # print("u", u)
        # u[2] = np.deg2rad(i)
        # wrap yaw command
        # if u[2] > np.pi:
        #     u[2] -= 2 * np.pi
        # elif u[2] < -np.pi:
        #     u[2] += 2 * np.pi
        x_next = model.rk45(x_current, u, dt, use_numeric=True,
                            wind=wind)
        t_current += dt
        state_history.append(x_next)
        time_history.append(t_current)
        x_current = x_next.copy()

    history_overall.append(np.array(state_history))

# Convert state history to a NumPy array for easier slicing
# state_history = np.array(state_history)  # Shape: (num_steps+1, 10)

plt.close()
# plot the color as a gradient of the yaw command

# set color pallete as a gradient of the yaw command
colors = sns.color_palette("hsv", len(psi_cmds))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i, history in enumerate(history_overall):
    x = history[:, 0]
    y = history[:, 1]
    z = history[:, 2]
    ax.plot(x, y, z, color=colors[i],
            label='cmd yaw: ' + str(np.rad2deg(psi_cmds[i])))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(x[0], y[0], z[0], c='g', marker='o')
    # set z bounds
    ax.set_zlim(-50, 50)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.scatter(x[0], y[0], z[0], c='g', marker='o')
# set z bounds
ax.legend()
ax.set_zlim(-50, 50)

state_history = np.array(history_overall[0])


# # Plot results
plt.figure(figsize=(12, 8))

# Plot airspeed (v is at index 6)
plt.subplot(2, 2, 1)
plt.plot(time_history, state_history[:, 6], 'b-', label='Airspeed (v)')
plt.xlabel('Time (s)')
plt.ylabel('Airspeed (m/s)')
plt.title('Airspeed vs. Time')
plt.legend()

# Plot x-position (x is at index 0)
plt.subplot(2, 2, 2)
plt.plot(time_history, state_history[:, 0], 'r-', label='x Position')
plt.xlabel('Time (s)')
plt.ylabel('x Position (m)')
plt.title('x Position vs. Time')
plt.legend()

# Plot yaw angle (psi is at index 5)
plt.subplot(2, 2, 3)
plt.plot(time_history, np.rad2deg(
    state_history[:, 5]), 'g-', label='Yaw (psi)')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (rad)')
plt.title('Yaw vs. Time')
plt.legend()

# Plot z-position (z is at index 2)
plt.subplot(2, 2, 4)
plt.plot(time_history, state_history[:, 2], 'm-', label='z Position')
plt.xlabel('Time (s)')
plt.ylabel('z Position (m)')
plt.title('z Position vs. Time')
plt.legend()

# plot the controls
fig, ax = plt.subplots(2, 2)

plt.tight_layout()


plt.show()
