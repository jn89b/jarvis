
import gc
import ray
import matplotlib.pyplot as plt
import pathlib
import torch
import numpy as np
import pickle as pkl
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
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.mask import ActionMaskingRLModule
from jarvis.utils.trainer import RayTrainerSimpleEnv

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


def create_hrl_env(config: Dict[str, Any],
                   env_config: Dict[str, Any]) -> HRLMultiAgentEnv:

    return HRLMultiAgentEnv(
        config=env_config)


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
    
# Function to compute saliency for a given logit
def compute_saliency(selected_logit, torch_obs_batch, policy):
    # Zero gradients in the policy and observation tensors
    policy.zero_grad()
    for tensor in torch_obs_batch.values():
        if tensor.grad is not None:
            tensor.grad.zero_()
    # Backpropagate from the selected logit.
    # Use retain_graph=True if you plan to perform multiple backward passes on the same graph.
    selected_logit.backward(retain_graph=True)
    # Collect gradients for each feature (here, we take the absolute value)
    saliency = {}
    for key, tensor in torch_obs_batch.items():
        # Assuming tensor shape [1, feature_dim]
        grad = tensor.grad.data[0].detach().numpy()  # shape: [feature_dim]
        saliency[key] = np.abs(grad)
    return saliency


def compute_saliency_map(env: PursuerEvaderEnv,
                         policy: SimpleEnvMaskModule,
                         observations: Dict[str, np.array]) -> None:
    """
    
    """
    import copy
    #action_logits = action_logits.detach().numpy().squeeze()
    torch_obs = {k: torch.tensor(np.array([v]), dtype=torch.float32, requires_grad=True)
                 for k, v in observations.items()}

    # Forward pass: obtain action logits
    action_logits = policy.forward_inference({"obs": copy.deepcopy(torch_obs)})["action_dist_inputs"]  # shape: [1, total_actions]
    
    if not torch_obs:
        raise ValueError("torch_obs is empty")
    
    # split the action logits into the roll, pitch, and yaw
    n_roll, n_alt, n_vel = env.action_spaces["0"]["action"].nvec
    
    roll_logits = action_logits[:, :n_roll]
    pitch_logits = action_logits[:, n_roll:n_roll+n_alt]
    vel_logits = action_logits[:, n_roll+n_alt:]
    
    #chosen_action_idx = torch.argmax(action_logits, dim=1).item()
    #selected_logit = action_logits[0, chosen_action_idx]
    
    best_roll = torch.argmax(roll_logits, dim=1).item()
    best_pitch = torch.argmax(pitch_logits, dim=1).item()
    best_vel = torch.argmax(vel_logits, dim=1).item()
    
    # get the logit of the best action
    roll_logit = roll_logits[0, best_roll]
    pitch_logit = pitch_logits[0, best_pitch]
    vel_logit = vel_logits[0, best_vel]
    
    # Backward pass: compute the gradient of the action logit with respect to the observation
    # to get the gradidents with respect to the observation
    # check if torch_obs is empty


    saliency_roll = compute_saliency(roll_logit, torch_obs, policy)
    saliency_pitch = compute_saliency(pitch_logit, torch_obs, policy)
    saliency_vel = compute_saliency(vel_logit, torch_obs, policy)

    # action_logits = action_logits.squeeze()
    # unwrapped_action: Dict[str, np.array] = env.unwrap_action_mask(
    #     action_logits)

    # discrete_actions = []
    # for k, v in unwrapped_action.items():
    #     best_action = torch.argmax(v)
    #     discrete_actions.append(best_action)
        


def infer(checkpoint_path: str, num_episodes: int = 1,
          use_pronav: bool = False, save: bool = False,
          index_save: int = 0, folder_dir: str = 'rl_pickle'):
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
    env.use_pronav = use_pronav
    # n_steps: int = 500
    # random seed
    # np.random.seed()
    # env.max_steps = 700
    print("max steps", env.max_steps)
    reward_list = []
    while not terminated['__all__']:
        import time
        start_time = time.time()
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
        end_time = time.time()
        print("time: ", end_time - start_time)
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
            # print("reward: ", reward)
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
        # print("data: ", i)
        print(data.phi)
        ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    print("env step", env.current_step)
    ax.set_xlabel('X Label (m)')
    ax.set_ylabel('Y Label (m)')
    ax.legend()
    # tight axis
    fig.tight_layout()

    # save the datas and the rewards
    pickle_info = {
        "datas": datas,
        "reward": reward
    }

    pickle_name = folder_dir+"/index_" + str(index_save) + "_reward.pkl"
    with open(pickle_name, 'wb') as f:
        pkl.dump(pickle_info, f)

    # fig, ax = plt.subplots()
    # reward_0 = [r['0'] for r in reward_list]
    # reward_1 = [r['1'] for r in reward_list]
    # reward_2 = [r['2'] for r in reward_list]
    # ax.plot(reward_0, label='Evader Reward')
    # ax.plot(reward_1, label='Pursuer Reward')
    # ax.plot(reward_2, label='Pursuer Reward 2')
    # print("sum of rewards: ", sum(reward_0), sum(reward_1))
    # ax.legend()
    # ray.shutdown()

    # plot a 3D plot of the agents
    # fig, ax = plt.subplots()
    # for i, data in enumerate(datas):
    #     print("data: ", i)
    #     ax.plot(data.v, label=f"Agent {i}")

    # ax.legend()
    # plt.show()


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

def load_and_infer_evader(checkpoint_path: str):
    """
    If I was to load this in real life:
    Need to load the purser policy 
    - Get the observation
    - Get the action mask
    - Compute a step in the environment and see what happens
    """
    # Load the pursuer policy from the checkpoint.
    evader_policy = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path) / "learner_group" /
        "learner" / "rl_module"
    )["evader_policy"]

    env = create_multi_agent_env(config=None, env_config=env_config)
    env.remove_all_agents()
    assert len(env.get_pursuer_agents()) == 0

    # insert an agent

    # evader_x: float = 100.0
    # evader_y: float = 0.0
    # evader_z: float = 40
    evader_x: float = np.random.uniform(-300, 300)
    evader_y: float = np.random.uniform(-300, 300)
    evader_z: float = np.random.uniform(20, 60)
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

    rand_x: float = np.random.uniform(-500, 500)
    rand_y: float = np.random.uniform(-500, 500)
    rand_z: float = np.random.uniform(35, 80)
    state_vector = StateVector(
        x=rand_x, y=rand_y, z=rand_z, yaw_rad=0.0, roll_rad=0,
        pitch_rad=0, speed=20)

    pursuer: Pursuer = Pursuer(
        agent_id="1",
        state_vector=state_vector,
        battle_space=env.battlespace,
        simple_model=PlaneKinematicModel(),
        is_controlled=True,
        radius_bubble=5
    )

    rand_x: float = np.random.uniform(-500, 500)
    rand_y: float = np.random.uniform(-500, 500)
    rand_z: float = np.random.uniform(35, 80)
    state_vector = StateVector(
        x=rand_x, y=rand_y, z=rand_z, yaw_rad=0.0, roll_rad=0,
        pitch_rad=0, speed=20)

    pursuer_2: Pursuer = Pursuer(
        agent_id="2",
        state_vector=state_vector,
        battle_space=env.battlespace,
        simple_model=PlaneKinematicModel(),
        is_controlled=True,
        radius_bubble=5
    )

    pursuer.capture_radius = 20
    env.insert_agent(evader)
    env.insert_agent(pursuer)
    env.insert_agent(pursuer_2)

    env.init_action_space()
    env.init_observation_space()
    # n_steps: int = 200

    terminated = {'__all__': False}
    # env.max_steps = 1000
    while not terminated['__all__']:
        # get number of age
        num_actions: int = env.action_spaces["0"]["action"].nvec.sum()
        current_evader: Evader = env.get_evader_agents()[0]
        obs = env.observe(current_evader, num_actions)
        torch_obs_batch = {k: torch.from_numpy(
            np.array([v])) for k, v in obs.items()}
        action_logits = evader_policy.forward_inference({"obs": torch_obs_batch})[
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
        action_dict['0'] = {'action': discrete_actions}
        observation, reward, terminated, truncated, info = env.step(
            action_dict=action_dict)

        # check if done
        if (terminated['__all__'] == True):
            print("reward: ", reward)
            break

    datas: List[DataHandler] = []
    agents = env.get_all_agents
    for agent in agents:
        agent: SimpleAgent
        data: DataHandler = agent.simple_model.data_handler
        datas.append(data)

    # plot a 3D plot of the agents
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(datas):
        # ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    ax.scatter(evader_x, evader_y, evader_z, label="Evader Start")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    # plt.show()


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
    evader_x: float = 100.0
    evader_y: float = 0.0
    evader_z: float = 40
    # evader_x: float = np.random.uniform(-300, 300)
    # evader_y: float = np.random.uniform(-300, 300)
    # evader_z: float = np.random.uniform(20, 60)
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

    rand_x: float = np.random.uniform(-500, 500)
    rand_y: float = np.random.uniform(-500, 500)
    rand_z: float = np.random.uniform(35, 80)
    state_vector = StateVector(
        x=rand_x, y=rand_y, z=rand_z, yaw_rad=0.0, roll_rad=0,
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
    env.max_steps = 1000
    env.use_pronav = False
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
        # ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    ax.scatter(evader_x, evader_y, evader_z, label="Evader Start")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    # plt.show()


def load_good_guy(checkpoint_path: str, index_save: int = 0,
                  folder_dir: str = '') -> None:
    """
    """
    ray.init(ignore_reinit_error=True)
    env = create_hrl_env(config=None, env_config=env_config)

    policies: MultiRLModuleSpec = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path) /
        "learner_group" / "learner" / "rl_module"
    )

    print("policies", policies.keys())

    good_guy_hrl = policies["good_guy_hrl"]
    good_guy_offensive = policies["good_guy_offensive"]
    good_guy_defensive = policies["good_guy_defensive"]
    pursuer = policies["pursuer"]

    observation, info = env.reset()
    terminated = {'__all__': False}
    # n_steps: int = 500
    # random seed
    # np.random.seed()
    # env.max_steps = 700
    print("max steps", env.max_steps)
    reward_list = []
    goal_history: List[StateVector] = []
    high_level_action_history: List[int] = []
    while not terminated['__all__']:
        current_action_history: List[int] = []
        # for i in range(n_steps):
        # compute the next action from a batch of observations
        # torch_obs_batch = torch.from_numpy(np.array([obs]))
        key_value = list(observation.keys())[0]

        if key_value == 'good_guy_hrl':
            obs = observation['good_guy_hrl']
            torch_obs_batch = {k: torch.from_numpy(
                np.array([v])) for k, v in obs.items()}
            action_logits = good_guy_hrl.forward_inference(
                {"obs": torch_obs_batch})["action_dist_inputs"]
        elif key_value == 'good_guy_offensive':
            obs = observation['good_guy_offensive']
            torch_obs_batch = {k: torch.from_numpy(
                np.array([v])) for k, v in obs.items()}
            action_logits = good_guy_offensive.forward_inference(
                {"obs": torch_obs_batch})["action_dist_inputs"]
        elif key_value == 'good_guy_defensive':
            obs = observation['good_guy_defensive']
            torch_obs_batch = {k: torch.from_numpy(
                np.array([v])) for k, v in obs.items()}
            action_logits = good_guy_defensive.forward_inference(
                {"obs": torch_obs_batch})["action_dist_inputs"]
        else:
            obs = observation[key_value]
            torch_obs_batch = {k: torch.from_numpy(
                np.array([v])) for k, v in obs.items()}
            action_logits = pursuer.forward_inference(
                {"obs": torch_obs_batch})["action_dist_inputs"]
        end_time = time.time()
        # For my action space I have a multidscrete environment
        # Since my action logits are a [1 x total_actions] tensor
        # I need to get the argmax of the tensor
        action_logits = action_logits.detach().numpy().squeeze()
        if key_value != 'good_guy_hrl':
            unwrapped_action: Dict[str, np.array] = env.unwrap_action_mask(
                action_logits)
            discrete_actions = []
            for k, v in unwrapped_action.items():
                v = torch.from_numpy(v)
                best_action = torch.argmax(v).numpy()
                discrete_actions.append(best_action)
        else:
            discrete_actions = [torch.argmax(
                torch.from_numpy(action_logits)).numpy()]
            discrete_actions = discrete_actions[0]

        # action = torch.argmax(action_logits).numpy()
        action_dict = {}
        action_dict[key_value] = {'action': discrete_actions}
        # print("action dict: ", action_dict)

        observation, reward, terminated, truncated, info = env.step(
            action_dict=action_dict)

        if key_value == 'good_guy_hrl':
            current_action_history.append(discrete_actions)

        # check if done
        if (terminated['__all__'] == True):
            print("reward: ", reward)
            target = env.target
            print("target: ", target.x, target.y, target.z)
            goal_history.append(target)
            break
        


    datas: List[DataHandler] = []
    agents = env.get_all_agents
    # agents
    new_agents = []
    new_agents.append(env.get_evader_agents()[0])
    new_agents.extend(env.get_pursuer_agents())
    agents = new_agents
    for agent in agents:
        data: DataHandler = agent.simple_model.data_handler
        datas.append(data)

    # plot a 3D plot of the agents
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(datas):
        ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
        ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

    target: StateVector = env.target
    # plot the goal target as a cylinder
    ax.scatter(target.x, target.y, target.z,
               label="Target", color='red', s=100)

    print("env step", env.current_step)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    # tight axis
    fig.tight_layout()
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.plot(current_action_history, label='High Level Action')
    ax.legend()
    

    # save the datas and the rewards
    # pickle_info = {
    #     "datas": datas,
    #     "reward": reward,
    #     "goal_history": goal_history
    # }

    # # save the datas and the rewards
    # pickle_name = "index_hrl" + str(index_save) + "_reward.pkl"
    # with open(pickle_name, 'wb') as f:
    #     pkl.dump(pickle_info, f)

    # save the datas and the rewards
    pickle_info = {
        "datas": datas,
        "reward": reward
    }

    pickle_name = folder_dir+"/index_" + str(index_save) + "_reward.pkl"
    with open(pickle_name, 'wb') as f:
        pkl.dump(pickle_info, f)


def run_multiple_sims(checkpoint_path: str, num_sims: int = 10,
                      type: str = 'evader',
                      save: bool = False,
                      use_random_seed: bool = True,
                      num_random_seeds: int = 10) -> None:

    if use_random_seed:
        for j in range(num_random_seeds):
            seed_num = j
            np.random.seed(seed_num)
            folder_name: str = 'hrl_data/'+'seed_'+str(seed_num)
            for i in range(num_sims):
                if type == 'pursuer_evader':
                    infer(checkpoint_path=checkpoint_path, num_episodes=1,
                          use_pronav=False, save=save, index_save=i,
                          folder_dir=folder_name)
                if type == 'pursuer':
                    load_and_infer_pursuer(checkpoint_path=checkpoint_path)
                if type == "evader":
                    load_and_infer_evader(checkpoint_path=checkpoint_path)
                if type == "good_guy":
                    load_good_guy(
                        checkpoint_path=checkpoint_path, index_save=i,
                        folder_dir=folder_name)

    else:
        np.random.seed(5)
        folder_dir = 'pursuer_evader_data_test'
        for i in range(num_sims):
            if type == 'pursuer_evader':
                infer(checkpoint_path=checkpoint_path, num_episodes=1,
                      use_pronav=False, save=save, index_save=i)
            if type == 'pursuer':
                load_and_infer_pursuer(checkpoint_path=checkpoint_path)
            if type == "evader":
                load_and_infer_evader(checkpoint_path=checkpoint_path)
            if type == "good_guy":
                load_good_guy(checkpoint_path=checkpoint_path, index_save=i,
                              folder_dir=folder_dir)

    plt.show()


if __name__ == '__main__':
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_13-57-07/PPO_pursuer_evader_env_5dfbc_00000_0_2025-02-17_13-57-07/checkpoint_000001"
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_14-19-01/PPO_pursuer_evader_env_6d682_00000_0_2025-02-17_14-19-02/checkpoint_000004"
    # path: str = "/home/justin/ray_results/PPO_2025-02-17_15-47-23/PPO_pursuer_evader_env_c517d_00000_0_2025-02-17_15-47-23/checkpoint_000070"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_03-05-25/PPO_pursuer_evader_env_a81fe_00000_0_2025-02-19_03-05-25/checkpoint_000074"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_11-13-19/PPO_pursuer_evader_env_d0af9_00000_0_2025-02-19_11-13-19/checkpoint_000071"
    # path: str = "/home/justin/ray_results/PPO_2025-02-19_22-19-31/PPO_pursuer_evader_env_e2123_00000_0_2025-02-19_22-19-31/checkpoint_000063"
    # path: str = "/home/justin/ray_results/PPO_2025-02-24_03-59-11/PPO_pursuer_evader_env_ff294_00000_0_2025-02-24_03-59-11/checkpoint_000006"
    path: str = "/home/justin/ray_results/PPO_2025-03-11_01-00-20/PPO_hrl_env_1d8d6_00000_0_2025-03-11_01-00-20/checkpoint_000131"
    # path:str = "/home/justin/ray_results/PPO_2025-03-10_19-56-29/PPO_hrl_env_ab204_00000_0_2025-03-10_19-56-30/checkpoint_000018"
    path:str = "/home/justin/ray_results/pursuer_evader_2/PPO_2025-02-24_13-25-45/PPO_pursuer_evader_env_24ee9_00000_0_2025-02-24_13-25-45/checkpoint_000224"
    #path:str = "/root/ray_results/PPO_2025-03-20_17-11-07/PPO_high_speed_pursuer_evader_3902d_00000_0_2025-03-20_17-11-07/checkpoint_000224"
    path:str = "/root/ray_results/PPO_2025-03-21_11-49-49/PPO_high_speed_pursuer_evader_80f0c_00000_0_2025-03-21_11-49-49/checkpoint_000061"
    path:str = "/root/ray_results/PPO_2025-03-21_11-49-49/PPO_high_speed_pursuer_evader_80f0c_00000_0_2025-03-21_11-49-49/checkpoint_000224"
    path:str = "/home/justin/ray_results/PPO_2025-03-26_01-05-00/PPO_pursuer_evader_env_40a11_00000_0_2025-03-26_01-05-00/checkpoint_000139"
    # ---- Pursuer Evader----
    path:str = "/root/ray_results/PPO_2025-03-28_10-46-27/PPO_pursuer_evader_env_cf49e_00000_0_2025-03-28_10-46-27/checkpoint_000015"
    path:str = "/root/ray_results/PPO_2025-03-28_10-46-27/PPO_pursuer_evader_env_cf49e_00000_0_2025-03-28_10-46-27/checkpoint_000055"
    path:str = "/home/justin/ray_results/PPO_2025-03-29_01-03-41/PPO_pursuer_evader_env_9075a_00000_0_2025-03-29_01-03-41/checkpoint_000007"
    # ---- HRL ----
    # path: str = "/home/justin/ray_results/PPO_2025-02-28_02-55-49/PPO_hrl_env_cecd1_00000_0_2025-02-28_02-55-50/checkpoint_000000"
    # plt.show()

    run_multiple_sims(checkpoint_path=path, num_sims=2, type='pursuer_evader',
                      use_random_seed=False)
    # ray_trainer = RayTrainerSimpleEnv(
    #     config_file="config/simple_env_config.yaml"
    # )
    # ray_trainer.infer_pursuer_evader(
    #     checkpoint_path=path, num_episodes=1, save=True,
    # )
    # ray_trainer.infer_multiple_times(checkpoint_path=path, 
    #                                  folder_name='pursuer_evader_high_speed_data_test',
    #                                  num_sims=5,
    #                                  use_random_seed=False, 
    #                                  type='pursuer_evader', 
    #                                  use_pronav = True ,
    #                                  save=True,
    #                                  start_count=0)
    