
import gc
import ray
import pathlib
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from typing import Dict, Any
from jarvis.utils.trainer import Trainer, load_yaml_config
from jarvis.envs.simple_multi_env import EngageEnv
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.mask import SimpleEnvMaskModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from ray.rllib.core.rl_module import RLModule
# from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
#     ActionMaskingTorchRLModule,
# )

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# A simple RL module that does not perform any action masking.

import torch.nn as nn


class SimpleTorchRLModule(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        # Build a basic fully connected network.
        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        # Simply pass the observations to the internal model.
        logits, _ = self.internal_model({"obs": input_dict["obs"]})
        return logits, state

    def value_function(self):
        return self.internal_model.value_function()

# from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
#     ActionMaskingTorchRLModule)

# tune.register_env("env", HierarchicalEnv)
# tune.register_env("mask_env", ActionMaskingHRLEnv)
# Register the custom model
# ModelCatalog.register_custom_model(
#     "masked_action_model", MultiActionMaskingTorchRLModule)


# Register your custom model with a name so you can reference it in your RLlib config.
# ModelCatalog.register_custom_model(
#     "unpacked_masked_torch_model", UnpackedMaskedActionsTorchModel)

gc.collect()

# Used to clean up the Ray processes after training
ray.shutdown()
# For debugging purposes
# ray.init(local_mode=True)
ray.init()


def create_env(config: Dict[str, Any],
               env_config: Dict[str, Any]) -> EngageEnv:
    engage_env = EngageEnv(
        config=env_config)

    return engage_env


def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:

    return PursuerEvaderEnv(
        config=env_config)


def create_hrl_agent_env(config: Dict[str, Any],
                         env_config: Dict[str, Any]) -> HRLMultiAgentEnv:

    return HRLMultiAgentEnv(
        config=env_config)


def train_rllib() -> None:
    """
    """
    # tune.register_env("env", HierarchicalEnv)

    env_config = load_yaml_config(
        "config/simple_env_config.yaml")['battlespace_environment']

    env_config = load_yaml_config(
        "config/simple_env_config.yaml")['battlespace_environment']


    tune.register_env("engage_env", lambda config:
                      create_env(config=config,
                                 env_config=env_config))

    example_env = create_env(config=None, env_config=env_config)

    if example_env.simulation_config is None:
        raise ValueError("simulation_config is None")

    observation_space = example_env.observation_space
    action_space = example_env.action_space
    # Define the run configuration for Ray Tune.
    run_config = tune.RunConfig(
        stop={"training_iteration": 1000},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=5,
            # checkpoint_score_attribute="episode_reward_mean",
            # checkpoint_score_order="max",
        ),
    )

    # Define the multi-agent RL training setup
    config = (
        PPOConfig()
        # .environment(env="env")
        # use create_env to pass in the env_config
        # .environment(env="engage_env")
        .environment(env="engage_env")
        .env_runners(num_env_runners=2)
        .resources(num_gpus=1)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=SimpleEnvMaskModule,
                observation_space=observation_space,
                action_space=action_space,
                model_config={},  # Additional model config options (if any)
            )
        )
        .env_runners(observation_filter="MeanStdFilter")
    )
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()
    ray.shutdown()


def train_multi_agent() -> None:

    def policy_mapping_fn(agent_id, episode, **kwargs):
        # Convert agent_id to int if it's not already\
        if int(agent_id) < 1:
            return "evader_policy"
        else:
            return "pursuer_policy"

    env_config = load_yaml_config(
        "config/simple_env_config.yaml")['battlespace_environment']

    tune.register_env("pursuer_evader_env", lambda config:
                      create_multi_agent_env(config=config,
                                             env_config=env_config))

    example_env = create_multi_agent_env(config=None, env_config=env_config)
    num_evaders = example_env.agent_config['num_evaders']

    if example_env.simulation_config is None:
        raise ValueError("simulation_config is None")

    # Define the run configuration for Ray Tune.
    run_config = tune.RunConfig(
        stop={"training_iteration": 4500},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=5,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
        ),
    )

    evader_obs_space = example_env.observation_spaces['0']
    evader_action_space = example_env.action_spaces['0']
    pursuer_obs_space = example_env.observation_spaces['1']
    pursuer_action_space = example_env.action_spaces['1']
    
    # Build the PPO configuration.
    config = (
        PPOConfig()
        # Use the registered normalized env.
        .environment(env="pursuer_evader_env")
        .framework("torch")
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "evader_policy": RLModuleSpec(
                        module_class=SimpleEnvMaskModule,
                        observation_space=evader_obs_space,
                        action_space=evader_action_space,
                        model_config={}
                    ),
                    "pursuer_policy": RLModuleSpec(
                        module_class=SimpleEnvMaskModule,
                        observation_space=pursuer_obs_space,
                        action_space=pursuer_action_space,
                        model_config={}
                    )
                }
            )
        )
        .multi_agent(
            policies={
                "evader_policy": (None, evader_obs_space, evader_action_space, {}),
                "pursuer_policy": (None, pursuer_obs_space, pursuer_action_space, {})
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(num_gpus=1)
        .env_runners(observation_filter="MeanStdFilter",
                     num_env_runners=12)
    )

    # Initialize and run the training using Ray Tune.
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()
    ray.shutdown()


def train_hrl(checkpoint_path=None) -> None:

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id == "good_guy_hrl":
            return "good_guy_hrl"
        if agent_id == "good_guy_offensive":
            return "good_guy_offensive"
        elif agent_id == "good_guy_defensive":
            return "good_guy_defensive"
        else:
            # All remaining agents (e.g. pursuers) use the pursuer policy.
            return "pursuer"

    env_config = load_yaml_config(
        "config/simple_env_config.yaml")['battlespace_environment']

    tune.register_env("hrl_env", lambda config:
                      create_hrl_agent_env(config=config,
                                           env_config=env_config))

    example_env = create_hrl_agent_env(config=None, env_config=env_config)
    example_env.close()

    # load the model from our checkpoints
    # # Create only the neural network (RLModule) from our checkpoint.
    if checkpoint_path is None:
        evader_policy = None
    else:
        evader_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )
        
        print("evader_policy", evader_policy)

    # Extract the observation and action spaces from your environment.
    # (You might need to create or compute these from HRLMultiAgentEnv)
    hrl_obs_space = example_env.observation_spaces["good_guy_hrl"]
    hrl_act_space = example_env.action_spaces["good_guy_hrl"]

    offensive_obs_space = example_env.observation_spaces["good_guy_offensive"]
    offensive_act_space = example_env.action_spaces["good_guy_offensive"]

    defensive_obs_space = example_env.observation_spaces["good_guy_defensive"]
    defensive_act_space = example_env.action_spaces["good_guy_defensive"]

    # assuming pursuer agent id '1'
    pursuer_obs_space = example_env.observation_spaces["1"]
    pursuer_act_space = example_env.action_spaces["1"]

    # Define the run configuration for Ray Tune.
    run_config = tune.RunConfig(
        stop={"training_iteration": 4500},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=5,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
        ),
    )

    # Build the PPO configuration.
    config = (
        PPOConfig()
        # Use the registered normalized env.
        .environment(env="hrl_env")
        .framework("torch")
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "good_guy_hrl": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=hrl_obs_space,
                        action_space=hrl_act_space,
                        model_config={}
                    ),
                    "good_guy_offensive": RLModuleSpec(
                        module_class=SimpleEnvMaskModule,
                        observation_space=offensive_obs_space,
                        action_space=offensive_act_space,
                        model_config={}
                    ),
                    "good_guy_defensive": RLModuleSpec(
                        module_class=SimpleEnvMaskModule,
                        observation_space=defensive_obs_space,
                        action_space=defensive_act_space,
                        model_config={}
                    ),
                    "pursuer": RLModuleSpec(
                        module_class=SimpleEnvMaskModule,
                        observation_space=pursuer_obs_space,
                        action_space=pursuer_act_space,
                        model_config={}
                    )
                }
            )
        )
        .multi_agent(
            policies={
                "good_guy_hrl": (None, hrl_obs_space, hrl_act_space, {}),
                "good_guy_offensive": (None, offensive_obs_space, offensive_act_space, {}),
                "good_guy_defensive": (evader_policy, defensive_obs_space, defensive_act_space, {}),
                "pursuer": (None, pursuer_obs_space, pursuer_act_space, {})
            },
            # policies=["good_guy_hrl", "good_guy_offensive",
            #           "good_guy_defensive", "pursuer"],
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["good_guy_hrl", "good_guy_offensive",
                               "good_guy_defensive", "pursuer"]
        )
        .resources(num_gpus=1)
        .env_runners(observation_filter="MeanStdFilter",
                     num_env_runners=10)
    )
    # Initialize and run the training using Ray Tune.
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()
    ray.shutdown()


if __name__ == '__main__':
    # main()
    # train_rllib()
    #train_multi_agent()
    #path:str = "/home/justin/ray_results/PPO_2025-04-20_13-25-48_evade/PPO_pursuer_evader_env_e216c_00000_0_2025-04-20_13-25-49/checkpoint_000014"
    #path: str = "/home/justin/ray_results/pursuer_evader_2/PPO_2025-02-24_13-25-45/PPO_pursuer_evader_env_24ee9_00000_0_2025-02-24_13-25-45/checkpoint_000224"
    path:str = "/home/justin/coding_projects/Jarvis/checkpoint_000048"
    train_hrl(checkpoint_path=path)
    ray.shutdown()
