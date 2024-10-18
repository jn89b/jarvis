import yaml
import os
from typing import List, Tuple, TYPE_CHECKING, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from jarvis.envs.env import DynamicThreatAvoidance, AbstractBattleEnv
from jarvis.utils.callbacks import SaveVecNormalizeCallback


def load_yaml_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


class Trainer():
    """
    """

    def __init__(self,
                 model_config: dict,
                 env_config: dict,
                 training_config: dict) -> None:

        # Model Config
        self.model_config: dict = model_config
        self.env_config: dict = env_config
        self.training_config: dict = training_config

        self.model_name: str = model_config.get("model_name", "Test")
        self.model_path: str = model_config.get("model_path", None)
        self.load_model: bool = model_config.get("load_model", False)
        self.vec_env_path: str = model_config.get("vec_env_path", None)
        self.save_dir: str = './models/'+self.model_name

        # Environment Config
        self.use_discrete_actions: bool = model_config.get(
            "use_discrete_actions", True)
        self.config_file_dir: str = env_config.get("config_file_dir", None)
        self.aircraft_config_dir: str = env_config.get(
            "aircraft_config_dir", None)
        self.pursuer_config_dir: str = env_config.get(
            "pursuer_config_dir", None)

        # Training Config
        self.total_time_steps: int = training_config.get(
            "total_time_steps", 1000000)
        self.save_freq: int = training_config.get("save_freq", 10000)
        self.upload_norm_obs: bool = training_config.get(
            "upload_norm_obs", False)
        self.num_envs: int = training_config.get("num_envs", 1)
        self.set_seed: bool = training_config.get("set_seed", False)
        self.seed: int = training_config.get("seed", 0)
        self.continue_training: bool = training_config.get(
            "continue_training", False)

    def save_config(self, save_path: str) -> None:
        """
        Saves the model, environment, and training configurations as a YAML file.
        """
        config = {
            'model_config': self.model_config,
            'env_config': self.env_config,
            'training_config': self.training_config,
            'aircraft_config': self.env_config.get("aircraft_config", {}),
            'pursuer_config': self.env_config.get("pursuer_config", {}),
            'config': self.env_config.get("config", {})
        }

        config_file = os.path.join(save_path, f"{self.model_name}_config.yaml")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(config_file, 'w') as file:
            yaml.dump(config, file)

        print(f"Configuration saved to {config_file}")

    def create_env(self) -> DynamicThreatAvoidance:
        env = DynamicThreatAvoidance(
            upload_norm_obs=self.upload_norm_obs,
            vec_env=self.vec_env_path,
            use_discrete_actions=self.use_discrete_actions,
            config_file_dir=self.config_file_dir,
            aircraft_config_dir=self.aircraft_config_dir,
            pursuer_config_dir=self.pursuer_config_dir
        )

        return env

    def train(self) -> None:
        # env = SubprocVecEnv([self.create_env for _ in range(self.num_envs)])
        # # env = self.create_env()
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)
        # env = VecMonitor(env)
        env = DummyVecEnv([lambda: self.create_env()])

        # test_env = self.create_env()
        # check_env(test_env)
        self.save_config(self.save_dir)

        callback = SaveVecNormalizeCallback(
            save_freq=self.save_freq,  # Save every 10,000 steps
            save_path=self.save_dir,
            name_prefix=self.model_name,
            vec_normalize_env=env,
            verbose=1
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.save_dir,
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
        else:
            print("Training model from scratch")
            model = PPO('MlpPolicy',
                        env,
                        verbose=1,
                        tensorboard_log="./logs/"+self.model_name,
                        learning_rate=0.0003,
                        device='cuda')
            model.learn(total_timesteps=self.total_time_steps,
                        log_interval=1,
                        callback=[callback, checkpoint_callback])
            model.save(self.save_dir)

        print("Training complete")
