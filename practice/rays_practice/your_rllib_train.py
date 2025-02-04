import ray
from ray import train, tune
from your_rllib_trainer import get_ppo_config
from ray.rllib.algorithms.ppo import PPOConfig
from your_rllib_environment import YourEnvironment
from your_rllib_config import get_multiagent_policies, policy_map_fn

ray.init(
    local_mode=True,
)

RUN_WITH_TUNE = True
NUM_ITERATIONS = 500

config: PPOConfig = get_ppo_config()
config.environment(
    env=YourEnvironment,
)

if RUN_WITH_TUNE:

    config.training(
        lr=tune.grid_search([0.01, 0.001, 0.0001]),
    )
    config.multi_agent(
        policies=get_multiagent_policies(),
        policy_mapping_fn=policy_map_fn,
        policies_to_train=list(get_multiagent_policies().keys()),
        count_steps_by="env_steps",
        observation_fn=None,
    )
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"env_runners/episode_return_mean": 150.0},
        ),
    )
    tuner.fit()

# else:
