import argparse
import gymnasium as gym
from gymnasium.spaces import Dict, Tuple, Box, Discrete
import os
import ray
import ray.tune as tune
from ray.tune.registry import register_env
# from ray.rllib.examples.env.nested_space_repeat_after_me_env import \
#     NestedSpaceRepeatAfterMeEnv
from ray.rllib.examples.envs.classes.nested_space_repeat_after_me_env import \
    NestedSpaceRepeatAfterMeEnv
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.0,
    help="Reward at which we stop training.")

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None)
    register_env("NestedSpaceRepeatAfterMeEnv",
                 lambda c: NestedSpaceRepeatAfterMeEnv(c))

    config = {
        "env": "NestedSpaceRepeatAfterMeEnv",
        "env_config": {
            "space": Dict({
                "a": Tuple(
                    [Dict({
                        "d": Box(-10.0, 10.0, ()),
                        "e": Discrete(2)
                    })]),
                "b": Box(-10.0, 10.0, (2, )),
                "c": Discrete(4)
            }),
        },
        "entropy_coeff": 0.00005,  # We don't want high entropy in this Env.
        "gamma": 0.0,  # No history in Env (bandit problem).
        "lr": 0.0005,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_sgd_iter": 4,
        "num_workers": 0,
        "vf_loss_coeff": 0.01,
        "framework": args.framework,
        "model": {
            "use_attention": True,
            "attention_num_transformer_units": 1,
            "attention_dim": 64,
            "attention_num_heads": 1,
            "attention_head_dim": 30,
            "attention_memory_inference": 50,
            "attention_memory_training": 50,
            "attention_position_wise_mlp_dim": 32,
            "attention_init_gru_gate_bias": 2.0,
            "attention_use_n_prev_actions": 15,
            "attention_use_n_prev_rewards": 15,
        },
    }

    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    results = tune.run(args.run, config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    print("Best trial so far: ", results.get_best_trial("episode_reward_mean",
                                                        mode="max"))
    ray.shutdown()