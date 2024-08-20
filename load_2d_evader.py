import matplotlib.pyplot as plt
import copy 
from stable_baselines3 import PPO
from jarvis.envs.simple_2d_env import BattleEnv
from jarvis.visualizer.visualizer import Visualizer

if __name__ == '__main__':
    model_name = "PPO_evader_2D_480000_steps"
    # model_name ="PPO_evader_2D_40000_steps"
    environment = BattleEnv()  # Create a single instance of the environment for evaluation
    model = PPO.load(model_name, env=environment)
    print("Model loaded.")
    
    #set seed number for reproducibility
    seed = 0
    num_times = 5
    num_success = 0
    battle_space_list = []
    reward_list = []
    idx_fail = []
    for i in range(num_times):
        obs, _ = environment.reset()
        done = False
        count = 0
        count_max = 350
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info  = environment.step(action)
            environment.render()
            count += 1
            if done or count >= count_max:
                if count >= count_max:
                    print("Success", count, i)
                    num_success += 1
                    done = True
                    battle_space_list.append(copy.deepcopy(environment.battlespace))
                else:
                    print("Failure", count, i)
                    idx_fail.append(i)
                    battle_space_list.append(copy.deepcopy(environment.battlespace))
                # print("Done: ", done)
                # print("Reward: ", reward)
                # print("Info: ", info)
    
    print(f"Success rate: {num_success/num_times}")

    data_vis = Visualizer()
    battlespace = environment.battlespace
    for i, battle_space in enumerate(battle_space_list):
        if i in idx_fail:
            fig, ax = data_vis.plot_2d_trajectory(battle_space)
            # fig, ax = data_vis.plot_attitudes2d(battle_space)
            pass
        else:
            fig, ax = data_vis.plot_2d_trajectory(battle_space)
            #fig, ax = data_vis.plot_attitudes2d(battle_space)

    plt.show()    

