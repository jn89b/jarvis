import matplotlib.pyplot as plt
import numpy as np
from jarvis.envs.simple_agent import PlaneKinematicModel
# Instantiate the PlaneKinematicModel with dt=0.1 and default tau values.
model = PlaneKinematicModel(dt_val=0.05)
plt.close()
# Define the initial state vector:
# Order: [x, y, z, phi, theta, psi, v, p, q, r]
# Start all at zero.
x0 = np.zeros(7)
x0[2] = 50
x0[-1] = 15
# Define a constant (step) control input:
# Order: [u_phi, u_theta, u_psi, v_cmd]
# Here, we set the attitude commands to 0 and command a step in airspeed to 10 m/s.
roll_cmd: float = np.deg2rad(0)
pitch_cmd: float = np.deg2rad(-20)
yaw_cmd: float = np.deg2rad(0)
u = np.array([roll_cmd, pitch_cmd, yaw_cmd, 20.0])

# Define wind as zero (i.e. [wind_x, wind_y, wind_z])
wind = np.array([10.0, 0.0, 0.0])

# Set simulation parameters
num_steps = 10
dt = model.dt_val

# Initialize storage for simulation data
state_history = [x0]
time_history = [0.0]

# Set current state and time
x_current = x0.copy()
t_current = 0.0

# Simulation loop: perform 50 integration steps using RK45
for i in range(num_steps):
    # print("u", u)
    # u[2] = np.deg2rad(i)
    # wrap yaw command
    if u[2] > np.pi:
        u[2] -= 2 * np.pi
    elif u[2] < -np.pi:
        u[2] += 2 * np.pi
    x_next = model.rk45(x_current, u, dt, use_numeric=True,
                        wind=wind)
    t_current += dt
    state_history.append(x_next)
    time_history.append(t_current)
    x_current = x_next.copy()

# Convert state history to a NumPy array for easier slicing
state_history = np.array(state_history)  # Shape: (num_steps+1, 10)

plt.close()
# Plot results
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
plt.plot(time_history, state_history[:, 5], 'g-', label='Yaw (psi)')
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

plt.tight_layout()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = state_history[:, 0]
y = state_history[:, 1]
z = state_history[:, 2]
ax.plot(x, y, z)

# set z bounds
ax.set_zlim(-50, 50)

plt.show()
