import numpy as np
import matplotlib.pyplot as plt

from jarvis.assets.Plane import Agent,Evader, Pursuer
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.visualizer.visualizer import Visualizer


def initialize_battle_space():
    battlespace = BattleSpace(
        np.array([-1000, 1000]),
        np.array([-1000, 1000]),
        np.array([-1000, 1000]),
        []
    )
    return battlespace


battlespace = initialize_battle_space()
state_vector = StateVector(
    0, 100, 50, 
    np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),
    20)

evader = Evader(
    battle_space=battlespace,
    state_vector=state_vector,
    id = 0,
    radius_bubble=10)

missle = Pursuer(
    battle_space=battlespace,
    state_vector=StateVector(
        0, 0, 20, 
        np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),
        20
    ),
    id = 1,
    radius_bubble=10)

battlespace.agents = [missle,evader]

N = 250
init_time = 0
distance_history = []
CAUGHT = False
for i in range(N):
    for agent in battlespace.agents:
        agent: Agent 
        print("agent id: ", np.rad2deg(agent.state_vector.yaw_rad))
        print("\n")
        if not agent.is_pursuer:
            agent.act({
                'yaw_cmd': np.deg2rad(-10),
                'pitch_cmd': np.deg2rad(0),
                'roll_cmd': np.deg2rad(-20),
                'speed_cmd': 25
            })
            distance = agent.state_vector.distance_3D(missle.state_vector)
            print("Distance: ", distance)
            distance_history.append(distance)
            if distance <= 15:
                CAUGHT = True
                print("Evader caught by missle")
                break
            
    if CAUGHT:
        break
    
    battlespace.step(0.1)
    init_time += 0.1
    
print("Final time: ", init_time)

data_vis = Visualizer()
fig, ax = data_vis.plot_3d_trajectory(battlespace)
fig, ax = data_vis.plot_attitudes(battlespace)

fig, ax = plt.subplots()
ax.plot(distance_history)

plt.show()