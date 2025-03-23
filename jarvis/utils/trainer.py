import yaml
import os
import matplotlib.pyplot as plt
import sys
import matplotlib
import json
import torch
import numpy as np
import pathlib
import ray
import pickle as pkl

from jarvis.utils.callbacks import SaveVecNormalizeCallback
from jarvis.envs.env import DynamicThreatAvoidance, AbstractBattleEnv
from jarvis.envs.agent import Agent, Evader, Pursuer

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO

from aircraftsim.utils.report_diagrams import SimResults
from typing import List, Tuple, TYPE_CHECKING, Dict, Any
from copy import deepcopy
from dataclasses import dataclass

from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.envs.simple_agent import DataHandler, Pursuer, Evader
from jarvis.utils.vector import StateVector
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.multi_agent_env import PursuerEvaderEnv


from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule

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
            "ego": current_ego_state,  # stores the x,y,z,theta,vx,vy of the evader
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


class RayTrainerSimpleEnv():
    def __init__(self,
                 convert_json: bool = True,
                 config_file:str="config/simple_env_config.yaml") -> None:
        # self.model_config: dict = model_config
        # self.env_config: dict = env_config
        # self.training_config: dict = training_config
        # Load your environment configuration (same as used in training).
        self.env_config = load_yaml_config(
            config_file)['battlespace_environment']
        self.convert_json = convert_json

    def train(self) -> None:
        pass

    def generate_dataset(self) -> None:
        pass

    def infer_multiple_times(self,
            checkpoint_path: str,
            folder_name:str, 
            num_sims: int = 10,
            type: str = 'evader',
            save: bool = False,
            use_random_seed: bool = True,
            num_random_seeds: int = 10,
            use_pronav:bool = True,
            start_count:int=0) -> None:
        
        if use_random_seed:
            for j in range(num_random_seeds):
                seed_num = j
                np.random.seed(seed_num)
                folder_name: str = 'hrl_data/'+'seed_'+str(seed_num)
                for i in range(num_sims):
                    if type == 'pursuer_evader':
                        index_save: int = int(i + start_count)
                        self.infer_pursuer_evader(checkpoint_path=checkpoint_path, 
                            num_episodes=1,
                            use_pronav=use_pronav, save=save, index_save=index_save,
                            folder_dir=folder_name)
                    # if type == 'pursuer':
                    #     load_and_infer_pursuer(checkpoint_path=checkpoint_path)
                    # if type == "evader":
                    #     load_and_infer_evader(checkpoint_path=checkpoint_path)
                    # if type == "good_guy":
                    #     load_good_guy(
                    #         checkpoint_path=checkpoint_path, index_save=i,
                    #         folder_dir=folder_name)

        else:
            np.random.seed(2)
            for i in range(num_sims):
                index_save: int = int(i + start_count)
                if type == 'pursuer_evader':
                    self.infer_pursuer_evader(checkpoint_path=checkpoint_path, 
                        num_episodes=1,
                        use_pronav=use_pronav, save=save, 
                        index_save=index_save,
                        folder_dir=folder_name)
                # if type == 'pursuer':
                #     load_and_infer_pursuer(checkpoint_path=checkpoint_path)
                # if type == "evader":
                #     load_and_infer_evader(checkpoint_path=checkpoint_path)
                # if type == "good_guy":
                #     load_good_guy(checkpoint_path=checkpoint_path, index_save=i,
                #                 folder_dir=folder_dir)
        
    def infer_pursuer_evader(
            self, checkpoint_path: str, 
            folder_dir:str, 
            num_episodes: int = 1,
            use_pronav: bool = True, save: bool = False,
            index_save: int = 0) -> None:

        ray.init(ignore_reinit_error=True)

        env = create_multi_agent_env(config=None, env_config=self.env_config)

        # load the model from our checkpoints
        # Create only the neural network (RLModule) from our checkpoint.
        evader_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["evader_policy"]

        pursuer_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["pursuer_policy"]

        episode_return = 0

        observation, info = env.reset()
        terminated = {'__all__': False}
        next_agent = observation.keys()
        env.use_pronav = use_pronav
        # n_steps: int = 500
        # random seed
        # np.random.seed()
        # env.max_steps = 700
        print("max steps", env.max_steps)
        reward_list = []
        while not terminated['__all__']:
            # for i in range(n_steps):
            # compute the next action from a batch of observations
            # torch_obs_batch = torch.from_numpy(np.array([obs]))
            key_value = list(observation.keys())[0]
            if key_value == '1':
                obs = observation['1']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
            elif key_value == '0':
                obs = observation['0']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = evader_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
            elif key_value == '2':
                obs = observation['2']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]

            # For my action space I have a multidscrete environment
            # Since my action logits are a [1 x total_actions] tensor
            # I need to get the argmax of the tensor
            action_logits = action_logits.detach().numpy().squeeze()
            unwrapped_action: Dict[str, np.array] = env.unwrap_action_mask(
                action_logits)

            discrete_actions = []
            for k, v in unwrapped_action.items():
                v = torch.from_numpy(v)
                best_action = torch.argmax(v).numpy()
                discrete_actions.append(best_action)

            # action = torch.argmax(action_logits).numpy()
            action_dict = {}
            action_dict[key_value] = {'action': discrete_actions}
            # print("action dict: ", action_dict)

            observation, reward, terminated, truncated, info = env.step(
                action_dict=action_dict)

            reward_list.append(reward)

            # check if done
            if (terminated['__all__'] == True):
                print("reward: ", reward)
                break

        datas: List[DataHandler] = []
        agents = env.get_all_agents
        for agent in agents:
            data: DataHandler = agent.simple_model.data_handler
            datas.append(data)

        folder_name = folder_dir+"/index_" + str(index_save)
        if self.convert_json:
            self.convert_to_json(datas, folder_name)

        # # plot a 3D plot of the agents
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for i, data in enumerate(datas):
        #     print("data: ", i)
        #     ax.scatter(data.x[0], data.y[1], data.z[2],
        #                label=f"Agent Start {i}")
        #     ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

        # print("env step", env.current_step)
        # ax.set_xlabel('X Label (m)')
        # ax.set_ylabel('Y Label (m)')
        # ax.legend()
        # # tight axis
        # fig.tight_layout()

        # # save the datas and the rewards
        # pickle_info = {
        #     "datas": datas,
        #     "reward": reward
        # }

        # pickle_name = folder_dir+"/index_" + str(index_save) + "_reward.pkl"
        # with open(pickle_name, 'wb') as f:
        #     pkl.dump(pickle_info, f)

    def convert_to_json(self, datas: List[DataHandler],
                        folder_name: str) -> None:
        """
        Converts data to be used to train predictformer
        """
        dataset_generator = DatasetGenerator(data_dir='./data')
        save_dir: str = folder_name+".json"
        evader_index: int = 0

        # pop out the evader
        evader_data: DataHandler = datas.pop(evader_index)
        simulation_data = []
        for i, x in enumerate(evader_data.x):
            time_step: float = i*0.1

            current_ego_state: List[float] = [
                evader_data.x[i],
                evader_data.y[i],
                evader_data.z[i],
                np.rad2deg(evader_data.phi[i]),
                np.rad2deg(evader_data.theta[i]),
                np.rad2deg(evader_data.psi[i]),
                evader_data.v[i]]

            current_control: List[float] = [
                0.0,  # we are controling yaw
                np.rad2deg(evader_data.theta[i]),
                np.rad2deg(evader_data.psi[i]),
                evader_data.v[i]]

            pursuers_states: List[List[float]] = []
            for pursuer in datas:
                pursuer_state: List[float] = [
                    pursuer.x[i], pursuer.y[i], pursuer.z[i],
                    np.rad2deg(pursuer.phi[i]),
                    np.rad2deg(pursuer.theta[i]),
                    np.rad2deg(pursuer.psi[i]),
                    pursuer.v[i]]

                pursuers_states.append(pursuer_state)

            data = dataset_generator.return_data(
                time_step, current_ego_state, current_control, pursuers_states)
            simulation_data.append(data)

        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(simulation_data, f, ensure_ascii=False, indent=4)


def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:

    return PursuerEvaderEnv(
        config=env_config)
