import numpy as np
import matplotlib.pyplot as plt

from jarvis.envs.simple_2d_env import EngagementEnv, BattleEnv, HRLBattleEnv
from jarvis.config import env_config_2d as env_config
from jarvis.visualizer.visualizer import Visualizer
from jarvis.envs.simple_2d_env import BattleEnv, HRLBattleEnv
from jarvis.visualizer.visualizer import Visualizer
from jarvis.utils.callbacks import SaveVecNormalizeCallback
from jarvis.config.env_config_2d import NUM_PURSUERS
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def create_env():
    evader_model_name = "PPO_evader_2D_1_pursuers_0"
    vec_evader_normalize_path = "PPO_evader_2D_1_pursuers_0.pkl"
    
    evader_env = DummyVecEnv([BattleEnv])
    evader_env = VecNormalize.load(vec_evader_normalize_path, evader_env)
    evader_env.training = False
    evader_env.norm_reward = False
    evader_model = PPO.load(evader_model_name, 
                            env=evader_env, print_system_info=True,
                            device='cuda')
    
    #return BattleEnv(spawn_own_space=False, spawn_own_agents=False)
    env = HRLBattleEnv(use_stable_baselines=True)
    return env 

if __name__ == '__main__':
    seed_num = 0
    # Create a list of environments to run in parallel
    num_envs = 4  # Adjust this number based on your CPU cores
    LOAD_MODEL = True
    CONTINUE_TRAINING = True
    COMPARE_MODELS = False
    TOTAL_TIME_STEPS = 4000000
    model_name = "PPO_hrl_2D"
    num_pursuers = str(NUM_PURSUERS) + '_pursuers'
    version_num = 1 
    full_model_name = model_name + '_' + num_pursuers + '_' + str(version_num)
    save_path = './models/' + full_model_name

    # Normalize the environment (observations and rewards)
    env = SubprocVecEnv([create_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)  # Monitor the vectorized environment
    
    # Check the environment to ensure it's correctly set up
    # test_env = BattleEnv()
    # check_env(test_env)
    # Add the callback

    # Define the CheckpointCallback to save the model 
    # every `save_freq` steps
    # Define the custom checkpoint callback
    checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=save_path,
        name_prefix=model_name,
        vec_normalize_env=env,
        verbose=1
    )
    
    # Load or initialize the model
    if LOAD_MODEL and not CONTINUE_TRAINING:
        #environment = BattleEnv()  # Create a single instance of the environment for evaluation
        model = PPO.load(full_model_name, env=env)
        print("Model loaded.")
    elif LOAD_MODEL and CONTINUE_TRAINING:
        print("Loading model and continuing training:", full_model_name)
        model = PPO.load(full_model_name, env=env)
        model.learn(total_timesteps=TOTAL_TIME_STEPS, 
                    log_interval=1,
                    callback=[checkpoint_callback])
        model.save(full_model_name)
        print("Model saved.")
    else:
        model = PPO('MlpPolicy', 
                    env, 
                    verbose=1,
                    tensorboard_log="./logs/"+full_model_name,
                    learning_rate=0.003,
                    ent_coef=0.01,
                    n_steps=2048,
                    batch_size=128,
                    seed=seed_num,
                    #use gpu
                    device='cuda')
        # Train the model in parallel
        model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[checkpoint_callback])
        
        model.save(model_name)
        print("Model saved.")