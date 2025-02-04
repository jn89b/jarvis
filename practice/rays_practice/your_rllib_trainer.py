from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.action_mask_env import ActionMaskEnv
from ray import air
from ray import tune

from ray.tune import Callback
# Define a custom callback to log rewards


class RewardLoggingCallback(Callback):
    def on_train_result(self, iteration, trials, trial, result, **info):
        # Extract the episode reward mean from the result
        episode_reward_mean = result.get("episode_reward_mean", None)
        if episode_reward_mean is not None:
            print(
                f"Iteration {iteration}: Episode Reward Mean = {episode_reward_mean}")


# import the cartpole env from the examples

def get_ppo_config() -> PPOConfig:
    config = PPOConfig()
    config.env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1
    )
    config.training(
        lr=0.0005,
        clip_param=0.2
    )

    return config
