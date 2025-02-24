
import gc
import ray
import matplotlib.pyplot as plt
import pathlib
import torch
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from typing import List, Dict, Any
from typing import Dict, Any
from jarvis.utils.trainer import Trainer, load_yaml_config
from jarvis.envs.simple_multi_env import EngageEnv
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.envs.simple_agent import DataHandler, Pursuer, Evader
from jarvis.utils.vector import StateVector
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)


from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from jarvis.envs.battlespace import BattleSpace


from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule
"""
Example script to load models from inference
https://docs.ray.io/en/latest/rllib/rllib-training.html
"""

plt.close('all')

gc.collect()

# Used to clean up the Ray processes after training
ray.shutdown()
# For debugging purposes
# ray.init(local_mode=True)
ray.init()

# https://github.com/ray-project/ray/issues/7983

# Assume these functions are defined as in your training code.
# For example:
#   - create_multi_agent_env(config=None, env_config=env_config)
#   - policy_mapping_fn(agent_id, episode, **kwargs)
#   - load_yaml_config(...)
#
# Here, we provide a dummy create_multi_agent_env for illustration.


def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:

    return PursuerEvaderEnv(
        config=env_config)


# Load your environment configuration (same as used in training).
env_config = load_yaml_config(
    "config/simple_env_config.yaml")['battlespace_environment']

# Define your policy mapping function as in training.


def policy_mapping_fn(agent_id, episode, **kwargs):
    # Here we assume agent IDs are integers.
    if int(agent_id) < 1:
        return "evader_policy"
    else:
        return "pursuer_policy"


# -------------------------
# Inference Code Starts Here
# -------------------------
tune.register_env("pursuer_evader_env", lambda config:
                  create_multi_agent_env(config=config,
                                         env_config=env_config))


def load_state(checkpoint_path: str, num_episodes: int = 1):
    ray.init(ignore_reinit_error=True)

    # Define your PPO configuration the same way as in training.
    # IMPORTANT: Make sure to use the same configuration parameters
    # that you used during training.
    from ray.rllib.algorithms.ppo import PPOConfig
    example_env = create_multi_agent_env(config=None, env_config=env_config)

    algo = Algorithm.from_checkpoint(checkpoint_path, env="pursuer_evader_env")
    ray.shutdown()


def infer(checkpoint_path: str, num_episodes: int = 1):
    ray.init(ignore_reinit_error=True)
    env = create_multi_agent_env(config=None, env_config=env_config)

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
    # n_steps: int = 500
    # random seed
    np.random.seed()
    env.max_steps = 700
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
        print("action dict: ", action_dict)

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

    # plot a 3D plot of the agents
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(datas):
        print("data: ", i)
        ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()

    fig, ax = plt.subplots()
    reward_0 = [r['0'] for r in reward_list]
    reward_1 = [r['1'] for r in reward_list]
    reward_2 = [r['2'] for r in reward_list]
    ax.plot(reward_0, label='Evader Reward')
    ax.plot(reward_1, label='Pursuer Reward')
    ax.plot(reward_2, label='Pursuer Reward 2')
    print("sum of rewards: ", sum(reward_0), sum(reward_1))
    ax.legend()
    ray.shutdown()

    # plot a 3D plot of the agents
    fig, ax = plt.subplots()
    for i, data in enumerate(datas):
        print("data: ", i)
        ax.plot(data.v, label=f"Agent {i}")

    ax.legend()
    plt.show()


# def infer_pursuer(checkpoint_path: str, num_episodes: int = 1):
#     """
#     Trying to see if I can load the pursuer policy and
#     take out a stationary target that is not moving

#     #TODO: Can I train a hierachial policy
#     that can choose between offense and defense?
#     """
#     ray.init(ignore_reinit_error=True)
#     env = create_multi_agent_env(config=None, env_config=env_config)

#     target_x: float = 250.0
#     target_y: float = 50.0

#     pursuer_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
#         pathlib.Path(checkpoint_path) /
#         "learner_group" / "learner" / "rl_module"
#     )["pursuer_policy"]

def load_and_infer_pursuer(checkpoint_path: str):
    """
    If I was to load this in real life:
    Need to load the purser policy 
    - Get the observation
    - Get the action mask
    - Compute a step in the environment and see what happens
    """
    # Load the pursuer policy from the checkpoint.
    pursuer_policy = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path) / "learner_group" /
        "learner" / "rl_module"
    )["pursuer_policy"]

    env = create_multi_agent_env(config=None, env_config=env_config)
    env.remove_all_agents()
    assert len(env.get_pursuer_agents()) == 0

    # insert an agent
    evader_x: float = 500.0
    evader_y: float = 0.0
    evader_z: float = 30
    state_vector = StateVector(
        x=evader_x, y=evader_y, z=evader_z, yaw_rad=0, roll_rad=0,
        pitch_rad=0, speed=0)
    evader: Evader = Evader(
        agent_id="0",
        state_vector=state_vector,
        battle_space=env.battlespace,
        simple_model=PlaneKinematicModel(),
        is_controlled=True,
        radius_bubble=5,
    )

    state_vector = StateVector(
        x=-200, y=500, z=50, yaw_rad=0.0, roll_rad=0,
        pitch_rad=0, speed=20)
    pursuer: Pursuer = Pursuer(
        agent_id="1",
        state_vector=state_vector,
        battle_space=env.battlespace,
        simple_model=PlaneKinematicModel(),
        is_controlled=True,
        radius_bubble=5
    )
    pursuer.capture_radius = 20
    env.insert_agent(evader)
    env.insert_agent(pursuer)

    env.init_action_space()
    env.init_observation_space()

    n_steps: int = 200

    terminated = {'__all__': False}
    env.max_steps = 10000
    while not terminated['__all__']:
        # get number of age
        num_actions: int = env.action_spaces["1"]["action"].nvec.sum()
        current_pursuer: Pursuer = env.get_pursuer_agents()[0]
        obs = env.observe(current_pursuer, num_actions)
        torch_obs_batch = {k: torch.from_numpy(
            np.array([v])) for k, v in obs.items()}
        action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"]
        evader.crashed = False
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
        action_dict['1'] = {'action': discrete_actions}
        observation, reward, terminated, truncated, info = env.step(
            action_dict=action_dict, specific_agent_id="1")

        # check if done
        if (terminated['__all__'] == True):
            print("reward: ", reward)
            break

    datas: List[DataHandler] = []
    agents = env.get_all_agents
    for agent in agents:
        agent: SimpleAgent
        if not agent.is_pursuer:
            continue
        data: DataHandler = agent.simple_model.data_handler
        datas.append(data)

    # plot a 3D plot of the agents
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(datas):
        print("data: ", i)
        ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    ax.scatter(evader_x, evader_y, evader_z, label="Evader Start")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_13-57-07/PPO_pursuer_evader_env_5dfbc_00000_0_2025-02-17_13-57-07/checkpoint_000001"
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_14-19-01/PPO_pursuer_evader_env_6d682_00000_0_2025-02-17_14-19-02/checkpoint_000004"
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_15-47-23/PPO_pursuer_evader_env_c517d_00000_0_2025-02-17_15-47-23/checkpoint_000070"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_03-05-25/PPO_pursuer_evader_env_a81fe_00000_0_2025-02-19_03-05-25/checkpoint_000074"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_11-13-19/PPO_pursuer_evader_env_d0af9_00000_0_2025-02-19_11-13-19/checkpoint_000071"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_22-19-31/PPO_pursuer_evader_env_e2123_00000_0_2025-02-19_22-19-31/checkpoint_000063"
    path: str = "/home/justin/ray_results/pursuer_evader/PPO_2025-02-22_16-12-02/PPO_pursuer_evader_env_0abe0_00000_0_2025-02-22_16-12-02/checkpoint_000414"
    infer(
        checkpoint_path=path, num_episodes=1)

    # load_and_infer_pursuer(checkpoint_path=path)
