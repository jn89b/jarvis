import matplotlib.pyplot as plt
import copy 
from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from jarvis.envs.simple_2d_env import BattleEnv
from jarvis.visualizer.visualizer import Visualizer
from jarvis.assets.Plane2D import Pursuer, Evader 
from jarvis.utils.Vector import StateVector
from jarvis.utils.math import normalize_obs, unnormalize_obs

import numpy as np
import torch
import random
import time 
import pickle as pkl
import gymnasium as gym

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def create_env():
    return BattleEnv()  # Adjust this to match your environment creation


class ExampleEnvironment(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3,))
        
        model_name = "PPO_evader_2D_280000_steps"
        vec_normalize_path = "PPO_evader_2D_vecnormalize_280000.pkl"

        env = DummyVecEnv([create_env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
        # Load the trained model
        self.model = PPO.load(model_name, env=env, print_system_info=True,
            device='cuda')
        
        self.environment = BattleEnv(upload_norm_obs=True, vec_env=env)
        self.obs, _ = self.environment.reset()

    def step(self, action=None):
        action,values = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, done, _, info = self.environment.step(action)
        return self.obs, reward, done, _ , info

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        else:
            seed = self.seed

        return self.environment.reset(seed=seed), 
    
    
if __name__ == '__main__':
    example = ExampleEnvironment()
    obs = example.reset(seed=0)
    done = False
    
    while not done:
        obs, reward, done, _, info = example.step()
        print("Reward: ", reward)
        # print("Info: ", info)
        # time.sleep(0.5)
        
    data_vis = Visualizer()
    battlespace = example.environment.battlespace
    fig, ax = data_vis.plot_2d_trajectory(battlespace)
    
    plt.show()
