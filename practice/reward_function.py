from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def normalize_angle(angle):
    """Normalize angle to the range [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def reward_heading_and_delta(heading_error: float,
                             delta_distance: float,
                             heading_error_max: float, delta_distance_max: float):
    """
    Computes a parabolic reward function scaled to [-1, 1].

    Parameters:
      heading_error     : The heading error in radians (desired - current).
                          (Should be normalized to [-pi, pi].)
      delta_distance    : The measured change in distance (current - previous).
                          A negative value means closing in.
      desired_delta     : The desired delta distance (a negative number).
      heading_error_max : The maximum acceptable heading error (radians) for normalization.
      delta_distance_max: The maximum acceptable error in delta distance for normalization.
                          (This is the difference between measured delta and desired delta.)

    Returns:
      reward : A scalar reward in the range [-1, 1] where:
               - 1 is best (zero heading error and perfect closing rate).
               - -1 is worst (errors at or beyond the maximum tolerances).

    The reward is defined as:
      E² = (heading_error / heading_error_max)² + ((delta_distance - desired_delta) / delta_distance_max)²
      reward = 1 - 2 * E², and then clipped to [-1, 1].
    """

    # Parabolic reward function: maximum reward of 1 when E=0, and -1 when E²=1.
    alpha = 1.2
    beta = -1
    delta_distance = delta_distance * 3
    distance_reward: float = np.exp(-alpha*delta_distance**2)
    heading_reward: float = (1.0 / (1.0 + np.exp(beta * heading_error)))
    reward: float = distance_reward * heading_reward

    return reward


heading_error = np.linspace(-np.pi, np.pi, 100)
delta_distance = np.linspace(-0.5, 0.5, 100)

reward_history = []
reward_evader = []
for h in heading_error:
    for d in delta_distance:

        reward = reward_heading_and_delta(heading_error=h,
                                          delta_distance=d,
                                          heading_error_max=np.pi,
                                          delta_distance_max=10)
        reward_history.append(reward)
        reward_evader.append(-reward)
        # print(
        #     f"Reward for heading error {h:.2f} and delta distance {d:.2f}: {reward:.2f}")


# heading_error = np.arange(-np.pi, np.pi, np.deg2rad(1))
# delta_distance = np.linspace(-5, 5, 100)

# reward_history = []

# for h in heading_error:
#     for d in delta_distance:

#         reward = reward_heading_and_delta(heading_error=h,
#                                           delta_distance=d,
#                                           heading_error_max=np.pi,
#                                           delta_distance_max=10)
#         reward_history.append(reward)
#         # print(
#         #     f"Reward for heading error {h:.2f} and delta distance {d:.2f}: {reward:.2f}")


# plot as a 3D plot
# k_vals = np.arange(0.1, 2, 0.1)

# overall_k = []
# for k in k_vals:
#     delta_tan_history = []
#     for i, d in enumerate(delta_distance):
#         val = np.tanh(k*d)
#         delta_tan_history.append(val)
#     overall_k.append(delta_tan_history)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(heading_error, delta_distance)
Z = np.array(reward_history).reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Heading Error')
ax.set_ylabel('Delta Distance')

# fig, ax = plt.subplots()
# for i, history in enumerate(overall_k):
#     k = k_vals[i]
#     ax.plot(delta_distance, history, label=f'k = {k}')
# ax.set_xlabel('Delta Distance')
# ax.set_ylabel('Tanh Delta Distance')
ax.legend()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(heading_error, delta_distance)
Z = np.array(reward_evader).reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Heading Error')
ax.set_ylabel('Delta Distance')


plt.show()
