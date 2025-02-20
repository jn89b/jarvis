from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
import numpy as np


class ProportionalNavigation3DHeuristicRLM(RLModule):
    """
    A heuristic RLModule that implements a 3D proportional navigation (PN) guidance law.
    It outputs commands for pitch, yaw, and velocity.

    Assumes each observation is a vector of shape (7,) containing:
      [rx, ry, rz, vx, vy, vz, current_speed]
    where:
      - (rx, ry, rz): relative position (from missile to target)
      - (vx, vy, vz): relative velocity (target relative to missile)
      - current_speed: the missile's current speed

    The PN law is implemented as:
         a_cmd = N * v_closing * (r x (v x r)) / ||r||^3

    The lateral acceleration a_cmd is then converted into pitch and yaw commands
    using small-angle approximations (assuming the missile’s velocity is aligned
    with its x-axis):
         pitch_command = arctan(-a_cmd_z / current_speed)
         yaw_command   = arctan(a_cmd_y / current_speed)

    The velocity command is set to a nominal value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = 3.0  # Navigation constant (tune as needed)
        self.nominal_velocity = 300.0  # Example nominal velocity in m/s

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        actions = []
        # Expect batch[Columns.OBS] to be an iterable of observations with shape (7,)
        for obs in batch[Columns.OBS]:
            rx, ry, rz, vx, vy, vz, current_speed = obs
            r = np.array([rx, ry, rz])
            v = np.array([vx, vy, vz])
            r_norm = np.linalg.norm(r)

            # Avoid division by zero
            if r_norm < 1e-6:
                actions.append(np.array([0.0, 0.0, current_speed]))
                continue

            # Compute closing velocity: negative projection of v on r
            closing_velocity = -np.dot(r, v) / r_norm

            # Compute the PN acceleration command in 3D:
            # a_cmd = N * closing_velocity * (r x (v x r)) / r_norm^3
            cross_v_r = np.cross(v, r)
            cross_r = np.cross(r, cross_v_r)
            a_cmd = self.N * closing_velocity * cross_r / (r_norm ** 3)

            # Convert the lateral acceleration components to pitch and yaw commands.
            # (Assumes missile is primarily moving along its x-axis.)
            yaw_command = np.arctan2(a_cmd[1], current_speed)
            pitch_command = np.arctan2(-a_cmd[2], current_speed)

            # Velocity command: here we simply command a nominal velocity.
            velocity_command = self.nominal_velocity

            actions.append(
                np.array([pitch_command, yaw_command, velocity_command]))

        return {Columns.ACTIONS: np.array(actions)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "ProportionalNavigation3DHeuristicRLM is not trainable! "
            "Make sure you do not include it in your `config.multi_agent(policies_to_train=...)` set."
        )

    @override(RLModule)
    def output_specs_inference(self):
        # Output: [pitch, yaw, velocity]
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]
