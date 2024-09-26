import numpy as np
import matplotlib.pyplot as plt

from jarvis.envs.simple_2d_env import EngagementEnv, ThreatAvoidEnv, HRLBattleEnv
from jarvis.config import env_config
from jarvis.visualizer.visualizer import Visualizer
from jarvis.envs.simple_2d_env import ThreatAvoidEnv
from jarvis.visualizer.visualizer import Visualizer

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

if __name__ == '__main__':
    evader_model_name = "PPO_evader_2D_280000_steps"
    vec_evader_normalize_path = "PPO_evader_2D_vecnormalize_280000.pkl"
    
    evader_env = DummyVecEnv([ThreatAvoidEnv])
    evader_env = VecNormalize.load(vec_evader_normalize_path, evader_env)
    evader_env.training = False
    evader_env.norm_reward = False
    evader_model = PPO.load(evader_model_name, 
                            env=evader_env, print_system_info=True,
                            device='cuda')
    
    #evader_env  = BattleEnv(upload_norm_obs=True, vec_env=evader_env)
    mean_rwd, std_reward = evaluate_policy(
        evader_model, evader_model.get_env(), n_eval_episodes=2,
        deterministic=True)
    
    env = HRLBattleEnv(use_stable_baselines=True)
    
    #check the environment
    check_env(env)
    steps = 1000
    random_number = np.random.randint(0, 100)
    env.reset(seed=2)
    # env.reset()
    n_times = 1
    reward_history = []
    for n in range(steps):
        action_dict = env.action_space.sample()
        obs, reward, done, _, info = env.step(action_dict)
        reward_history.append(reward)
        if done:
            break
        
    data_vis = Visualizer()
    avoid_battlespace = env.evade_env.battlespace
    target_battlespace = env.engage_env.battlespace
    fig, ax = data_vis.plot_2d_trajectory(avoid_battlespace)
    fig, ax = data_vis.plot_2d_trajectory(target_battlespace)
    target_location = target_battlespace.target.state_vector
    ax.plot(target_location.x, target_location.y, 'ro', label='Target')
    ax.legend() 
    
    # fig, ax = data_vis.plot_attitudes2d(avoid_battlespace)

    #plot the reward history
    fig, ax = plt.subplots()
    ax.plot(reward_history)

    plt.show()
        