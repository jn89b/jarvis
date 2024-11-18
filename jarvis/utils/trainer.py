from jarvis.utils.callbacks import SaveVecNormalizeCallback
from jarvis.envs.env import DynamicThreatAvoidance, AbstractBattleEnv
from jarvis.envs.agent import Agent, Evader, Pursuer

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO

from aircraftsim.utils.report_diagrams import SimResults
from typing import List, Tuple, TYPE_CHECKING, Dict
from copy import deepcopy
from dataclasses import dataclass
import yaml
import os
import matplotlib.pyplot as plt
import sys
import matplotlib
import json
matplotlib.use('TkAgg')


def load_yaml_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


@dataclass
class DataCache:
    evader: SimResults
    pursuers: List[SimResults]
    time_vector: List[float] = None


class DatasetGenerator():
    def __init__(self,
                 data_dir: str) -> None:
        self.data_dir: str = data_dir

    def return_data(self, time_step: float, current_ego_state: List[float],
                    current_control: List[float],
                    pursuers_states: List[List[float]]) -> None:
        return {
            "time_step": time_step,
            "ego": current_ego_state,  # stores the x,y,theta,vx,vy of the evader
            "controls": current_control,  # stores the controls of the evader
            # list of lists of rel_x,rel_y, rel_thet, rel_velocity of pursuers
            "vehicles": pursuers_states,
        }


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
        self.vec_env: VecNormalize = None
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
        env = SubprocVecEnv([self.create_env for _ in range(self.num_envs)])
        # env = self.create_env()
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        env = VecMonitor(env)

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

    def infer(self,
              vec_normal_path: str = None,
              num_evals: int = 10) -> None:
        """
        Load the model for evaluation
        """
        env = DummyVecEnv([self.create_env])
        # env = VecNormalize.load(self.vec_env_path, venv=env)

        if vec_normal_path is None:
            vec_normal_path: str = self.model_name + "_vecnormalize.pkl"
            env = VecNormalize.load(vec_normal_path, env)
            env.training = False
            env.norm_reward = False
            self.vec_env = env

        model = PPO.load(self.model_name, env=env, print_system_info=True,
                         device='cuda')

        env: DynamicThreatAvoidance = self.create_env()
        overall_reports: Dict[str, List[SimResults]] = {}
        for i in range(num_evals):
            obs, _ = env.reset()
            done = False
            while not done:
                action, values = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                if done:
                    reports: List[SimResults] = [
                        agent.sim_interface.report for agent in env.all_agents]
                    # envs.append(deepcopy(env))
                    if reward < 0:
                        name = "Failed"
                    else:
                        name = "Success"
                    overall_reports[name+str(i)] = reports
                    break

        for key, reports in overall_reports.items():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, individual_report in enumerate(reports):

                if i == 0:
                    label: str = 'Evader'
                else:
                    label: str = 'Pursuer' + str(i)
                ax.scatter(
                    individual_report.x[0], individual_report.y[0],
                    individual_report.z[0], label=label)
                ax.plot(individual_report.x, individual_report.y,
                        individual_report.z, label=label)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title("Run: " + key)

        plt.show()

    def generate_dataset(self,
                         vec_normal_path: str = None,
                         num_evals: int = 2,
                         success_only: bool = True,
                         dt_desired: float = 0.1) -> None:
        """
        Generates JSON dataset for the given model for each evaluatinon
        """
        env = DummyVecEnv([self.create_env])
        # env = VecNormalize.load(self.vec_env_path, venv=env)

        if vec_normal_path is None:
            vec_normal_path: str = self.model_name + "_vecnormalize.pkl"
            env = VecNormalize.load(vec_normal_path, env)
            env.training = False
            env.norm_reward = False
            self.vec_env = env

        model = PPO.load(self.model_name, env=env, print_system_info=True,
                         device='cuda')

        env: DynamicThreatAvoidance = self.create_env()

        overall_reports: Dict[str, List[SimResults]] = {
            'Success': [], 'Failed': []
        }
        for i in range(num_evals):
            obs, _ = env.reset()
            done = False
            # add a key to the dictionary
            while not done:
                action, values = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                if done:
                    pursuer_reports: List[SimResults] = []
                    for i, agent in enumerate(env.all_agents):
                        if i == 0:
                            ego_agent = agent.sim_interface.report
                        else:
                            pursuer_reports.append(agent.sim_interface.report)
                    data = DataCache(
                        evader=ego_agent, pursuers=pursuer_reports)
                    if reward < 0:
                        name = "Failed"
                    else:
                        name = "Success"
                    overall_reports[name].append(data)
                    break

        dataset_generator = DatasetGenerator(data_dir='./data')
        fz: int = env.get_freq()

        # if we want to generate data at 10 Hz
        dt = 1/fz
        dt_desired = 0.1
        fz_desired = 1/dt_desired

        """
        We will need to loop through each simulation
        - For each simulation, we will create a json file
        -
        """
        folder_dir = "./evader_data_1"
        idx: int = 0
        overall_reports = overall_reports['Success']
        print("Generating dataset", overall_reports)
        for cache in overall_reports:
            json_file_name: str = folder_dir + \
                "/simulation_" + str(idx) + ".json"
            simulation_data = []
            start_time: float = 0.0

            cache: DataCache
            evader: SimResults = cache.evader
            pursuers: List[SimResults] = cache.pursuers

            for j, x_val in enumerate(evader.x):
                time_step: float = start_time + j*dt
                if j % fz_desired != 0:
                    continue

                current_ego_state: List[float] = [
                    evader.x[j], evader.y[j], evader.z[j],
                    evader.roll_dg[j], evader.pitch_dg[j], evader.yaw_dg[j],
                    evader.airspeed[j]]

                current_control: List[float] = [
                    evader.roll_dg[j],
                    evader.pitch_dg[j],
                    evader.yaw_dg[j],
                    evader.airspeed[j]]

                pursuers_states: List[List[float]] = []
                for pursuer in pursuers:
                    pursuer_state: List[float] = [
                        pursuer.x[j], pursuer.y[j], pursuer.z[j],
                        pursuer.roll_dg[j], pursuer.pitch_dg[j], pursuer.yaw_dg[j],
                        pursuer.airspeed[j]]
                    pursuers_states.append(pursuer_state)

                data = dataset_generator.return_data(
                    time_step, current_ego_state, current_control, pursuers_states)
                simulation_data.append(data)

            # specify encoding
            with open(json_file_name, 'w', encoding='utf-8') as f:
                json.dump(simulation_data, f, ensure_ascii=False, indent=4)

            print(f"Data saved to {json_file_name}")
            idx += 1
            # plot the trajectory
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(evader.x, evader.y, evader.z, label='Evader')
            for pursuer in pursuers:
                ax.plot(pursuer.x, pursuer.y, pursuer.z, label='Pursuer')
            ax.set_xlabel('X')
            # save the image
            img_file_name = folder_dir + "/simulation_" + str(idx) + ".png"
            fig.savefig(img_file_name)

            # save a 2D plot
            fig, ax = plt.subplots()
            ax.plot(evader.x, evader.y, label='Evader')
            for pursuer in pursuers:
                ax.plot(pursuer.x, pursuer.y, label='Pursuer')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            img_file_name = folder_dir + "/simulation_" + str(idx) + "_2d.png"
            fig.savefig(img_file_name)

            # plt.show()
            # save the data to a json file
