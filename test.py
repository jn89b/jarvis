import numpy as np
import matplotlib.pyplot as plt
from aircraftsim import ReportGraphs, DataVisualizer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.env import DynamicThreatAvoidance
import torch
import copy
if __name__ == "__main__":
    dyn_env = DynamicThreatAvoidance()
    env_copy = copy.deepcopy(dyn_env)  # Try deepcopying the environment
    # print("observation space", dyn_env.observation_space)
    # print("action space", dyn_env.action_space)
    # check_env(dyn_env)

    # check cuda is available
    print("cuda available", torch.cuda.is_available())

    N = 4000
    for i in range(N):
        action: np.ndarray = dyn_env.action_space.sample()
        # print("action", action)
        obs, reward, terminated, _, info = dyn_env.step(action)
        # if terminated:
        #     break
        dyn_env.reset()

    reports = []
    x_list = []
    y_list = []
    z_list = []
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for agent in dyn_env.all_agents:

        reports.append(agent.sim_interface.report)
        x_list.append(agent.sim_interface.report.x)
        y_list.append(agent.sim_interface.report.y)
        z_list.append(agent.sim_interface.report.z)
        ax.plot(agent.sim_interface.report.x, agent.sim_interface.report.y,
                agent.sim_interface.report.z, label=f"Agent {agent.id}")
        ax.scatter(agent.sim_interface.report.x[-1], agent.sim_interface.report.y[-1],
                   agent.sim_interface.report.z[-1], label=f"Agent {agent.id} End")
        print("Agent", agent.id, "Final Position", agent.sim_interface.report.x[-1],
              agent.sim_interface.report.y[-1], agent.sim_interface.report.z[-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # plot velocities
    fig, ax = plt.subplots()
    for agent in dyn_env.all_agents:
        ax.plot(agent.sim_interface.report.airspeed, label=f"Agent {agent.id}")

    ax.legend()

    plt.show()
