import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.simple_2d_env import EngagementEnv
from jarvis.envs.battle_space_2d import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config
from jarvis.assets.Plane2D import Pursuer, Evader
from tests.utils import setup_battlespace
from jarvis.visualizer.visualizer import Visualizer
from stable_baselines3.common.env_checker import check_env


env = EngagementEnv(use_stable_baselines=True)
steps = 300
dt = env_config.DT
reward_history = []
n_times = 1
success = 0
for n in range(n_times):
    count = 0
    done = False
    #reset the environment
    env.reset()
    target = env.battlespace.target
    agent = env.agents[0]
    dx = target.state_vector.x - agent.state_vector.x
    dy = target.state_vector.y - agent.state_vector.y
    theta = np.arctan2(dy, dx)
    heading_error  = theta - agent.state_vector.yaw_rad
    reward_history = []
    for step in range(steps):
        # print("Step: ", step)
        dx = target.state_vector.x - agent.state_vector.x
        dy = target.state_vector.y - agent.state_vector.y
        theta = np.arctan2(dy, dx)
        
        heading_error  = theta - agent.state_vector.yaw_rad
        action_dict = env.action_space.sample()
        action_dict[0] = 1.0
        # action_dict[0] = -heading_error
        # action_dict[1] = 20
        
        obs, reward, done, _, info = env.step(action_dict, norm_action=True)
        print("obs: ", obs)
        reward_history.append(reward)
        if done or count >= steps:
            if reward > 0:
                success += 1
            else:
                print("Lose")
            break    

print(f"Success rate: {success/n_times}")
data_vis = Visualizer()
battlespace = env.battlespace
fig, ax = data_vis.plot_2d_trajectory(battlespace)
#PLOT THE target
target = battlespace.target
ax.plot(target.state_vector.x, target.state_vector.y, 'ro')
fig, ax = data_vis.plot_attitudes2d(battlespace)

#plot the reward history
fig, ax = plt.subplots()
ax.plot(reward_history)

plt.show()