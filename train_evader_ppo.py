import os
import ray
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from jarvis.envs.battle_env_single import BattleEnv
from jarvis.config import env_config
from ray.tune.registry import register_env

def env_creator(env_config=None):
    return BattleEnv()  # return an env instance

env_name = "single_agent_battle_env"
register_env(env_name, env_creator)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# Set the working directory in the Ray runtime environment
ray.init(runtime_env={"working_dir": "."})

if __name__ == "__main__":
    # Instantiate the environment to obtain observation and action spaces
    temp_env = BattleEnv()

    # Set the base PPO configuration
    base_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        #adjust hyperparameters
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .rollouts(num_envs_per_worker=5, 
                  num_rollout_workers=5)  # Number of environments per worker
        .training(
            #lr=tune.grid_search([3e-3]),
            lr = 3e-3,
            train_batch_size=8000,  # Example batch size
            
            # vf_clip_param= 50.0,
            # num_sgd_iter=10,
        )
    )

    # Set the policy with observation and action spaces
    base_config = base_config.to_dict()  # Convert PPOConfig to a dictionary
    base_config['multiagent'] = {
        "policies": {
            "default_policy": (None, temp_env.observation_space, temp_env.action_space, {})
        }
    }
    #normalize the observation space
    base_config['observation_filter'] = "MeanStdFilter"

    # Set up the experiment
    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)
    exp_name = "tune_analyzing_results"

    run_config = train.RunConfig(
        name = "PPO_Avoid_Single",
        storage_path=storage_path,
        log_to_file=True,
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True
        ),
        stop={"env_runners/episode_len_mean": env_config.MAX_NUM_STEPS,
              "timesteps_total": 2000000},
    )
    
    tuner = tune.Tuner(
        "PPO",
        param_space=base_config,
        # local_dir=storage_path,
        # log_to_file=True,
        run_config=run_config,
        )

    tuner.fit()

    # tune.run(
    #     "PPO",
    #     name="PPO_UAM",
    #     stop={"timesteps_total": 1500000},
    #     config=base_config,
    #     checkpoint_freq=20,
    #     checkpoint_at_end=True,
    #     storage_path=storage_path,
    #     log_to_file=True,
    # )
    
