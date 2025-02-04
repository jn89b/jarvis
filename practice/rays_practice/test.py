from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .training(
        lr=tune.grid_search([0.01, 0.001, 0.0001]),
    )
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(
        stop={"env_runners/episode_return_mean": 150.0},
    ),
)

tuner.fit()
