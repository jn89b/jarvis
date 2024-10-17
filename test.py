import numpy as np
import matplotlib.pyplot as plt
from aircraftsim import ReportGraphs, DataVisualizer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.env import DynamicThreatAvoidance

if __name__ == "__main__":
    dyn_env = DynamicThreatAvoidance()
    # print("observation space", dyn_env.observation_space)
    # print("action space", dyn_env.action_space)

    N = 1000
    for i in range(N):
        action: np.ndarray = dyn_env.action_space.sample()
        # print("action", action)
        dyn_env.step(action)
        # dyn_env.reset()

reports = []
x_list = []
y_list = []
z_list = []
for agent in dyn_env.all_agents:
    reports.append(agent.sim_interface.report)
    x_list.append(agent.sim_interface.report.x)
    y_list.append(agent.sim_interface.report.y)
    z_list.append(agent.sim_interface.report.z)


# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, report in enumerate(reports):
    ax.plot(report.x, report.y, report.z, label=f"Agent {i}")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
