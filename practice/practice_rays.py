import os
import ray
from gymnasium.spaces import Box, Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

from pettingzoo.utils import wrappers
from your_env_file import raw_env  # Import your environment file

torch, nn = try_import_torch()


### CUSTOM RLlib TORCH MODEL (Handles Action Masking) ###
class TorchMaskedActions(DQNTorchModel):
    """PyTorch model that supports action masking."""

    def __init__(self, obs_space: Box, action_space: Discrete, num_outputs, model_config, name, **kw):
        DQNTorchModel.__init__(self, obs_space, action_space,
                               num_outputs, model_config, name, **kw)

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(
                obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        """Handles action masking."""
        action_mask = input_dict["obs"]["action_mask"]

        # Compute predicted action logits
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]})

        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    ray.init()

    ### REGISTER ENVIRONMENT ###
    def env_creator():
        env = raw_env()  # Your custom PettingZoo environment
        return wrappers.ParallelEnv(env)  # Wrap as parallel env for RLlib

    env_name = "robber_police_game"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ### REGISTER CUSTOM MODEL ###
    ModelCatalog.register_custom_model("masked_dqn_model", TorchMaskedActions)

    ### CONFIGURE RLlib TRAINING ###
    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "masked_dqn_model"},
        )
        .multi_agent(
            policies={
                "prisoner": (None, obs_space, act_space, {}),
                "guard": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(log_level="DEBUG")
        .framework(framework="torch")
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,
            }
        )
    )

    ### TRAINING LOOP ###
    tune.run(
        "DQN",
        name="DQN_Training_RobberPolice",
        stop={"timesteps_total": 10000000 if not os.environ.get(
            "CI") else 50000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )
