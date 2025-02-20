
import gc
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from typing import Dict, Any
from jarvis.utils.trainer import Trainer, load_yaml_config
from jarvis.envs.simple_multi_env import EngageEnv
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.mask import SimpleEnvMaskModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog

# from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
#     ActionMaskingTorchRLModule,
# )


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
import gc

gc.collect()

# Used to clean up the Ray processes after training
ray.shutdown()
# For debugging purposes
# ray.init(local_mode=True)
ray.init()

#


def create_env(config: Dict[str, Any],
               env_config: Dict[str, Any]) -> EngageEnv:
    engage_env = EngageEnv(
        config=env_config)

    return engage_env


def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:

    return PursuerEvaderEnv(
        config=env_config)


def train_rllib() -> None:
    """
    """
    # tune.register_env("env", HierarchicalEnv)

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
        stop={"training_iteration": 1500},
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
                     num_env_runners=6)
    )

    # Initialize and run the training using Ray Tune.
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()
    ray.shutdown()


if __name__ == '__main__':
    # main()
    # train_rllib()
    train_multi_agent()
    ray.shutdown()
