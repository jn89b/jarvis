import os
import ray
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from jarvis.envs.battle_env import BattleEnv
from jarvis.config import env_config
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

# from ray.rllib.utils.check_env import check_env


# https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/different_spaces_for_agents.py

# Register your custom environment
def env_creator(env_config=None):
    return BattleEnv()  # return an env instance

def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "0":
        return "0"
    # else:
    #     return "1"

env_name = "battle_env"
register_env(env_name, env_creator)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
ray.init(runtime_env={"working_dir": "."})

if __name__ == "__main__":
    # Instantiate the environment to obtain observation and action spaces
    temp_env = BattleEnv()

    # Set the base PPO configuration
    base_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .rollouts(
            num_envs_per_worker=1, 
            num_rollout_workers=1
        )
        .multi_agent(
            policies = {"evader_policy": 
                (None, temp_env.observation_space[0], 
                 temp_env.action_space[0], {})},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: 
                "evader_policy" if 0 else "evader_policy",
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "evader_policy": SingleAgentRLModuleSpec(),
                }
            ),
        )
        # .training(
        #     lr=3e-3,
        #     # train_batch_size=8000,  # Example batch size
        #     # sgd_minibatch_size=256,
        #     # num_sgd_iter=10,
    )
    
    temp_env.reset()
    
    # Set the multi-agent configuration
    base_config = base_config.to_dict()  # Convert PPOConfig to a dictionary
    # base_config['multiagent'] = {
    #     "policies": {
    #         "evader_policy": (None, temp_env.observation_space[0], temp_env.action_space[0], {}),
    #         # "pursuer_policy": (None, temp_env.observation_space[1], temp_env.action_space[1], {}),
    #     },
    #     "policy_mapping_fn": lambda agent_id, episode, **kwargs: 
    #         "evader_policy" if 0 else "evader_policy",
    # }
    # base_config['multiagent'] = {
    #     "policies": {
    #         "evader_policy": (None, temp_env.observation_space[0], temp_env.action_space[0], {}),
    #         # "pursuer_policy": (None, temp_env.observation_spaces['pursuer'], temp_env.action_spaces['pursuer'], {}),
    #     },
    #     "policy_mapping_fn": lambda agent_id: "evader_policy",
    # }

    # Optional: Add custom reward and observation filters
    base_config['observation_filter'] = "MeanStdFilter"

    # Set up the experiment
    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)
    exp_name = "tune_analyzing_results"

    run_config = train.RunConfig(
        name = "PPO_BattleEnv",
        storage_path=storage_path,
        log_to_file=True,
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True
        ),
        stop={"timesteps_total": 1000000},
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=base_config,
        run_config=run_config,
    )

    tuner.fit()
