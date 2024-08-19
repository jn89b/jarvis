import unittest
import numpy as np
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
n_times = 20
success = 0
for n in range(n_times):
    count = 0
    done = False
    #reset the environment
    env.reset()
    target = env.battlespace.target
    dx = target.state_vector.x - env.battlespace.evader.state_vector.x
    dy = target.state_vector.y - env.battlespace.evader.state_vector.y
    
    for step in range(steps):
        # print("Step: ", step)
        action_dict = env.action_space.sample()
        obs, reward, done, _, info = env.step(action_dict)
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