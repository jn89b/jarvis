import matplotlib.pyplot as plt
import numpy as np
import copy
#from jarvis.envs.battle_env import BattleEnv
from jarvis.envs.battle_env_single import BattleEnv
from jarvis.visualizer.visualizer import Visualizer

if __name__ == "__main__":
    battle_env = BattleEnv()
    print(battle_env.action_space.sample())
    #random actions
    # control_dict = {
    #     0: np.array([0, 0, 0, 0]),
    # }
    
    #check size of observation space
    # print(battle_env.observation_spaces)
    
    USE_RANDOM_ACTIONS = True
    reward_history = []
    n_times = 1
    for n in range(n_times):
        battle_env.reset()
        for i in range(500):
            action_dict = battle_env.action_space.sample()
            print("Action Dict: ", action_dict)
            if not USE_RANDOM_ACTIONS:
                # action_dict[0][0] = np.random.uniform(-np.pi, np.pi)
                # action_dict[0][0] = np.deg2rad(35) #roll
                # action_dict[0][1] = np.deg2rad(10) #pitch
                # action_dict[0][2] = np.deg2rad(15) #yaw
                action_dict[0] = np.deg2rad(30)
                action_dict[1] = np.deg2rad(35) 
                action_dict[2] = np.deg2rad(10) #pitch
                action_dict[3] = np.deg2rad(15) #yaw
            #action_dict[0][3] = 25.0 #speed
                
            # battle_env.step(action_dict)
            obs, reward, done, info, _ = battle_env.step(action_dict)
            # print("Reward: ", reward)
            reward_history.append(reward)
            if done:
                print("Done: ", done)
                break

        # print("Done: ", done)
        # print("Info: ", info)
        # print("Obs: ", obs)
        # battle_env.render()    
        
    
    data_vis = Visualizer()
    battlespace = battle_env.battlespace
    fig, ax = data_vis.plot_3d_trajectory(battlespace)
    fig, ax = data_vis.plot_attitudes(battlespace)
    # plt.show()
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    plt.show()
    