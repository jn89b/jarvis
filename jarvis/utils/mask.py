import gymnasium as gym
from typing import Dict, Optional, Tuple, Union

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class ActionMaskingRLModule(RLModule):
    """An RLModule that implements action masking for safe RL.

    Expects the environment observation space to be a Dict with keys:
        - "observations": The raw observation (e.g. a Box).
        - "action_mask": A binary mask (e.g. a MultiBinary) for the discrete actions.
    """
    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "gym.spaces.Dict observation space with keys 'action_mask' and 'observations'."
            )
        self.observation_space_with_mask = observation_space
        # The underlying observation used by the network is stored in "observations".
        self.observation_space = observation_space["observations"]
        self._checked_observations = False

        # Call parent constructor with the raw observation space.
        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )


class MultiDimensionalMaskModule(ActionMaskingRLModule, PPOTorchRLModule, ValueFunctionAPI):
    """An RLModule for PPO with action masking that supports a multi-dimensional action mask.

    This module assumes the discrete action space is MultiDiscrete with three branches (roll, alt, vel)
    and that the environment provides an action mask as a MultiBinary tensor of shape (n_roll, n_alt, n_vel).
    """

    @override(PPOTorchRLModule)
    def setup(self):
        super().setup()
        # Reset the observation space to include the action mask.
        self.observation_space = self.observation_space_with_mask

    @override(PPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        outs = super()._forward_train(batch, **kwargs)
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        if isinstance(batch[Columns.OBS], dict):
            action_mask, batch = self._preprocess_batch(batch)
            batch["action_mask"] = action_mask
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        # Check that the batch observation includes the required keys.
        self._check_batch(batch)
        action_mask = batch[Columns.OBS].pop("action_mask")
        batch[Columns.OBS] = batch[Columns.OBS].pop("observations")
        return action_mask, batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> None:
        if not self._checked_observations:
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError(
                    "No action_mask found in observation. Ensure the observation space contains "
                    "'action_mask' and 'observations'."
                )
            if "observations" not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. Ensure the observation space contains "
                    "'action_mask' and 'observations'."
                )
            self._checked_observations = True

    def _mask_action_logits(
        self, outputs: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """Masks the action logits using the provided multi-dimensional action mask.

        The logits produced by the network are assumed to be a flattened vector for all branches,
        with total_actions = n_roll + n_alt + n_vel. We split these logits into three parts,
        apply the corresponding mask to each, and then concatenate them back.

        Assumes that the provided `action_mask` tensor is of shape (B, n_roll, n_alt, n_vel)
        (or (n_roll, n_alt, n_vel) if unbatched) and that it was generated as the outer product of
        independent masks for each branch.

        Returns:
            A dictionary with the modified logits under Columns.ACTION_DIST_INPUTS.
        """
        logits = outputs[Columns.ACTION_DIST_INPUTS]
        batch_size = logits.shape[0]
        n_roll, n_alt, n_vel = self.action_space.nvec  # e.g., [19, 100, 15]

        # Assume action_mask is a tensor of shape (B, total_actions) with total_actions = 134.
        # Split it into individual masks.
        roll_mask = action_mask[:, :n_roll]  # shape: (B, n_roll)
        alt_mask = action_mask[:, n_roll:n_roll+n_alt]  # shape: (B, n_alt)
        vel_mask = action_mask[:, n_roll+n_alt:]  # shape: (B, n_vel)

        # Convert the allowed masks to log-space.
        inf_mask_roll = torch.clamp(
            torch.log(roll_mask.float()), min=FLOAT_MIN)
        inf_mask_alt = torch.clamp(torch.log(alt_mask.float()), min=FLOAT_MIN)
        inf_mask_vel = torch.clamp(torch.log(vel_mask.float()), min=FLOAT_MIN)

        # Split logits into branches.
        roll_logits = logits[:, :n_roll]
        alt_logits = logits[:, n_roll:n_roll+n_alt]
        vel_logits = logits[:, n_roll+n_alt:]

        # Apply the log-space masks.
        roll_logits = roll_logits + inf_mask_roll
        alt_logits = alt_logits + inf_mask_alt
        vel_logits = vel_logits + inf_mask_vel

        # Concatenate the logits back.
        masked_logits = torch.cat([roll_logits, alt_logits, vel_logits], dim=1)
        outputs[Columns.ACTION_DIST_INPUTS] = masked_logits
        return outputs
