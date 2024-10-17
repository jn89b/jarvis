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

report: ReportGraphs = dyn_env.agents[0].sim_interface.report
data_vis = DataVisualizer(report)
data_vis.plot_3d_trajectory()
data_vis.plot_attitudes()
data_vis.plot_airspeed()
plt.show()
