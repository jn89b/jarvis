
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from typing import Dict, Any
from jarvis.utils.trainer import Trainer, load_yaml_config
from jarvis.envs.multi_env import TargetEngageEnv
from jarvis.utils.mask import MultiDimensionalMaskModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
# from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
#     ActionMaskingTorchRLModule,
# )

from jarvis.utils.mask import MultiDimensionalMaskModule
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


# For debugging purposes
ray.init(local_mode=True)

# tune.register_env("env", HierarchicalEnv)
tune.register_env("env", TargetEngageEnv)


def main() -> None:

    config_dir: str = "config/training_config.yaml"
    # Load the YAML file
    config = load_yaml_config(config_dir)

    # Access each configuration component
    model_config = config.get('model_config', {})
    env_config = config.get('env_config', {})
    training_config = config.get('training_config', {})
    trainer = Trainer(model_config=model_config,
                      env_config=env_config,
                      training_config=training_config)
    trainer.train()


def create_env(env_config: Dict[str, Any]) -> TargetEngageEnv:
    """
    """
    # env = HierarchicalEnv(env_config)
    config_env: str = "config/env_config.yaml"
    # self.aircraft_config_dir: str = env_config.get(
    # "aircraft_config_dir", "config/aircraft_config.yaml")
    aircraft_config_dir: str = "config/aircraft_config.yaml"

    env = TargetEngageEnv(
        battlespace=None,
        agents=None,
        upload_norm_obs=False,
        use_discrete_actions=True,
        config_file_dir=config_env,
        aircraft_config_dir=aircraft_config_dir,
    )

    return env


def train_rllib() -> None:
    """
    """

    # tune.register_env("env", HierarchicalEnv)
    tune.register_env("mask_env", create_env)
    example_env = create_env(env_config=None)

    observation_space = example_env.observation_space
    action_space = example_env.action_space
    # Define the run configuration for Ray Tune.
    run_config = tune.RunConfig(
        stop={"training_iteration": 100},
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
        .environment(env="env")
        .env_runners(num_env_runners=1)
        .resources(num_gpus=0)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=MultiDimensionalMaskModule,
                observation_space=observation_space,
                action_space=action_space,
                model_config={},  # Additional model config options (if any)
            )
        )
    )
    # .rl_module(
    #     # We need to explicitly specify here RLModule to use and
    #     # the catalog needed to build it.
    #     rl_module_spec=RLModuleSpec(
    #         module_class=MultiActionMaskingTorchRLModule,
    #         observation_space=observation_space,
    #         action_space=action_space,
    #         model_config={}
    #         # module_class=ActionMaskingTorchRLModule,
    #         # observation_space=observation_space,
    #         # action_space=action_space,
    #         # model_config={}
    #     ),
    # )

    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()


if __name__ == '__main__':
    # main()
    train_rllib()
