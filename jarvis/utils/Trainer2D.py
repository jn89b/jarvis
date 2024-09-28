from typing import List, Tuple, TYPE_CHECKING, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.simple_2d_env import EngagementEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from jarvis.utils.callbacks import SaveVecNormalizeCallback
from jarvis.envs.simple_2d_env import AbstractBattleEnv, AvoidThreatEnv, EngagementEnv, RCSEnv
# from jarvis.assets.Plane2D import Agent, Evader, Pursuer, Obstacle
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from jarvis.visualizer.visualizer import Visualizer
from matplotlib import pyplot as plt
import copy 

class RLTrainer2D():
    """
    This class is used to train whatever environment from jarvis.envs you want.
    env_name is the index of the environment you want to train.
    0 is AvoidThreatEnv
    1 is EngagementEnv
    2 is RCSEnv
    """
    
    def __init__(self, 
                 model_name:bool=None,
                 num_envs:int=5,
                 load_model:bool=True,
                 continue_training:bool=True,
                 total_time_steps:int=3000000,
                 spawn_own_space:bool=False,
                 spawn_own_agents:bool=False,
                 use_stable_baselines:bool=True,
                 upload_norm_obs:bool=False,
                 use_discrete_actions:bool=False,
                 vec_env:VecNormalize=None,
                 vec_env_path:str=None,
                 save_freq:bool=10000,
                 env_type:int=0) -> None:
        
        if model_name is None:
            raise ValueError("Model name cannot be None.")
        else:
            self.model_name = model_name
            
        self.num_envs = num_envs
        self.load_model = load_model
        self.continue_training = continue_training
        self.total_time_steps = total_time_steps
        self.env_type = env_type
        self.spawn_own_space = spawn_own_space
        self.spawn_own_agents = spawn_own_agents
        self.use_stable_baselines = use_stable_baselines
        self.upload_norm_obs = upload_norm_obs
        self.use_discrete_actions = use_discrete_actions
        self.vec_env = vec_env
        self.save_freq = save_freq
        self.vec_env_path = vec_env_path
        
    def create_env(self) -> AbstractBattleEnv:
        if self.env_type == 0:
            print("Creating AvoidThreatEnv")
            env = AvoidThreatEnv(
                spawn_own_space=self.spawn_own_space,
                spawn_own_agents=self.spawn_own_agents, 
                use_stable_baselines=self.use_stable_baselines,
                randomize_threats=True,
                randomize_start=False,
                use_discrete_actions=self.use_discrete_actions,
                vec_env=self.vec_env,
                upload_norm_obs=self.upload_norm_obs,
                )
        elif self.env_type == 1:
            env = EngagementEnv(
                spawn_own_space=self.spawn_own_space,
                spawn_own_agents=self.spawn_own_agents, 
                use_stable_baselines=self.use_stable_baselines,
                randomize_goal=True,
                randomize_start=False,
                use_discrete_actions=self.use_discrete_actions,
                vec_env=self.vec_env,
                upload_norm_obs=self.upload_norm_obs,
                )
        elif self.env_type == 2:
            env = RCSEnv(
                spawn_own_space=self.spawn_own_space,
                spawn_own_agents=self.spawn_own_agents, 
                use_stable_baselines=self.use_stable_baselines,
                use_discrete_actions=self.use_discrete_actions,
                vec_env=self.vec_env,
                upload_norm_obs=self.upload_norm_obs,
                )
        else:
            raise ValueError("Invalid environment type.")
            
        return env
            
    def train(self) -> None:
        env = SubprocVecEnv([self.create_env for _ in range(self.num_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        env = VecMonitor(env)

        test_env = self.create_env()
        check_env(test_env)
        
        print("model name:", self.model_name)
        callback = SaveVecNormalizeCallback(
            save_freq=self.save_freq,  # Save every 10,000 steps
            save_path='./models/'+self.model_name,
            name_prefix=self.model_name,
            vec_normalize_env=env,
            verbose=1   
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq, 
            save_path='./models/'+self.model_name,
            name_prefix=self.model_name
        )
        
        if self.load_model and self.continue_training:
            print("Loading model and continuing training:", self.model_name)
            vec_normal_path = self.model_name + "_vecnormalize.pkl"
            env = VecNormalize.load(vec_normal_path, venv=env)
            model = PPO.load(self.model_name, env=env, devce='cuda')
            model.learn(total_timesteps=self.total_time_steps, 
                        log_interval=1,
                        callback=[callback, checkpoint_callback])
            model.save(self.model_name)
            print("Model saved.")
        else:
            model = PPO('MlpPolicy', 
                        env, 
                        verbose=1,
                        tensorboard_log="./logs/"+self.model_name,
                        learning_rate=0.0003,
                        device='cuda')
            model.learn(total_timesteps=self.total_time_steps, 
                        log_interval=1,
                        callback=[callback, checkpoint_callback])
            model.save(self.model_name)
            print("Model saved.")
        
    def infer_model(self, num_evals:int=3,
                    vis_results:bool=True) -> None:
        
        env = DummyVecEnv([self.create_env])

        if self.vec_env is None:
            vec_normal_path:str = self.model_name + "_vecnormalize.pkl"
            env = VecNormalize.load(vec_normal_path, env)
            env.training = False
            env.norm_reward = False
            self.vec_env = env
            
        model = PPO.load(self.model_name, env=env, print_system_info=True,
                         device='cuda')
        # evaluate the policy
        mean_rwd, std_reward = evaluate_policy(
            model, model.get_env(), n_eval_episodes=10,
            deterministic=True)
        print("Mean reward and standard deviation:", mean_rwd, std_reward)
        print("\n")
        
        # environment = model.get_env()
        #reset the environment
        environment = self.create_env()
        
        battle_space_list = []
        reward_list = []
        idx_fail = []
        detection_overall = []
        success_history = []
        for i in range(num_evals):
            obs, _ = environment.reset()
            done = False
            count = 0
            detections = []
            while not done:
                action, values = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = environment.step(action)
                count += 1
                detections.append(info['detection_probability'])
                reward_list.append(reward)
                if done:
                    if reward > 0:
                        battle_space_list.append(
                            copy.deepcopy(environment.battlespace))
                        success_history.append(True)
                    else:
                        idx_fail.append(i)
                        battle_space_list.append(
                            copy.deepcopy(environment.battlespace))
                        success_history.append(False)
                    detection_overall.append(detections)
        
        win_percentage = sum(success_history) / num_evals
        print("win percentage:", win_percentage)
        
        data_vis = Visualizer()
        for i, battle_space in enumerate(battle_space_list):
            #fig, ax = data_vis.plot_2d_trajectory(battle_space)
            if success_history[i]:
                if self.env_type == 2:
                    fig, ax = data_vis.plot_radars(battle_space)
                    fig, ax = plt.subplots()
                    ax.plot(detection_overall[i])
                else:
                    fig,ax = data_vis.plot_2d_trajectory(battle_space)
                ax.set_title("Success")
            else:
                if self.env_type == 2:
                    fig, ax = data_vis.plot_radars(battle_space)
                else:
                    fig,ax = data_vis.plot_2d_trajectory(battle_space)
                    ax.plot(detection_overall[i])
                ax.set_title("Failure")
        
        # if self.env_type == 2:
        #     #plot the detections
            
        #     #visualize the radar
        #     fig, ax = data_vis.plot_radars(battle_space_list[0])
            
        #     #plot the trajectories over time
        #     for i, detections in enumerate(detection_overall):
        #         fig, ax = plt.subplots()
        #         ax.plot(detections)
        #         ax.set_title("Detections")
            
        
        #plot the rewards
        fig, ax = plt.subplots()
        ax.plot(reward_list)
        
        # plot the results
        plt.show()
        
        
        
    def test_environment(self) -> None:
        env = self.create_env()
        check_env(env)
        
        obs = env.reset()
        reward_history = []
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            # env.render()
            reward_history.append(reward)
            if done:
                obs = env.reset()
                
        data_vis = Visualizer()
        fig, ax = data_vis.plot_2d_trajectory(env.battlespace)
        
        fig, ax = plt.subplots()
        ax.plot(reward_history)
        
        plt.show()