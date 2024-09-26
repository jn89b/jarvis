import matplotlib.pyplot as plt
import copy 
from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from jarvis.envs.simple_2d_env import AvoidThreatEnv
from jarvis.visualizer.visualizer import Visualizer
from jarvis.assets.Plane2D import Pursuer, Evader 
from jarvis.utils.Vector import StateVector
from jarvis.utils.math import normalize_obs, unnormalize_obs

import numpy as np
import torch
import random
import time 
import pickle as pkl

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_env():
    return AvoidThreatEnv(
        
        )  # Adjust this to match your environment creation

if __name__ == '__main__':
    model_name = "PPO_evader_2D_3000000_steps"
    vec_normalize_path = "PPO_evader_2D_vecnormalize_3010000.pkl"
    USE_PICKLE_PURSUERS = True
    # Load the environment and normalization statistics
    num_envs = 5
    # Create a DummyVecEnv for single environment inference
    env = DummyVecEnv([create_env])
    
    if VecNormalize:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    
    # Load the trained model
    model = PPO.load(model_name, env=env, print_system_info=True,
                     device='cuda')
    print("Model loaded.")
    
    # Set seed number for reproducibility
    seed = 1
    set_global_seed(seed=seed)
    
    num_times = 5
    num_success = 0
    battle_space_list = []
    reward_list = []
    idx_fail = []

    # Evaluate the policy
    mean_rwd, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=2,
        deterministic=True)
    print("Mean reward and standard deviation:", mean_rwd, std_reward)
    print("\n")
    #environment = env.envs[0]  # Access the first environment from the VecNormalize wrapper
    environment = AvoidThreatEnv(
        upload_norm_obs=True, 
        vec_env=env,
        use_discrete_actions=True)
    
    # obs, _ = environment.reset(seed=seed)
    for i in range(num_times):
        obs, _ = environment.reset()    
        done = False
        count = 0
        count_max = environment.time_steps
        
        while not done:
            start_time = time.time()
            action, values = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = environment.step(action)
            end_time = time.time() - start_time
            # environment.render()
            count += 1
            reward_list.append(reward)
            
            if done or count >= count_max:
                if count >= count_max and reward > 0:
                    num_success += 1
                    done = True
                    battle_space_list.append(
                        copy.deepcopy(environment.battlespace))
                else:
                    idx_fail.append(i)
                    battle_space_list.append(
                        copy.deepcopy(environment.battlespace))

    win_percentage = num_success / num_times
    print(f"Win percentage: {win_percentage}")
    # Visualization
    data_vis = Visualizer()
    battlespace = environment.battlespace
    
    for i, battle_space in enumerate(battle_space_list):
        if i in idx_fail:
            fig, ax = data_vis.plot_2d_trajectory(battle_space)
            #fig, ax =data_vis.plot_attitudes2d(battle_space, ignore_pursuer=True)
            #set super title
            fig.suptitle(f"LOSE {i}")
        else:
            fig, ax = data_vis.plot_2d_trajectory(battle_space)
            # plot title
            #fig, ax =data_vis.plot_attitudes2d(battle_space, ignore_pursuer=True)
            fig.suptitle(f"WIN {i}")
    # Plot the rewards
    fig, ax = plt.subplots()
    ax.plot(reward_list)    
    plt.show()
