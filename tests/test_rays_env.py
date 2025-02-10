import unittest
import numpy as np
from typing import Dict, Any
from jarvis.envs.multi_env import TargetEngageEnv
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.vector import StateVector


class TestRaysEnv(unittest.TestCase):

    def setUp(self):
        self.env_config: Dict[str, Any] = load_yaml_config(
            "config/env_config.yaml")
        # self.aircraft_config_dir: str = env_config.get(
        # "aircraft_config_dir", "config/aircraft_config.yaml")
        self.aircraft_config_dir: Dict[str, Any] = load_yaml_config(
            "config/aircraft_config.yaml")
        self.environment = TargetEngageEnv(
            battlespace=None,
            agents=None,
            upload_norm_obs=False,
            use_discrete_actions=True,
            config_file_dir="config/env_config.yaml",
            aircraft_config_dir="config/aircraft_config.yaml",
        )

    def test_load_env(self) -> None:
        """
        Unit test to make sure the environment configuration is loaded correctly
        and that the environment actually captures the configuration
        """
        self.assertIsNotNone(self.env_config)
        self.assertIsNotNone(self.aircraft_config_dir)

        control_limits = self.aircraft_config_dir.get("control_limits")
        print(control_limits)
        self.assertIsNotNone(control_limits)

        # check to see if the environment has loaded an agent
        self.assertIsNotNone(self.environment.get_controlled_agents)
        self.assertIsNotNone(self.environment.battlespace)
        agents = self.environment.get_controlled_agents
        battlespace_agents = self.environment.battlespace.all_agents

        # assert that we only have one agent
        self.assertEqual(len(agents), 1)
        print("agents",  agents)
        # if this i
        print("battlespace agents", battlespace_agents)

    def test_target_spawn(self):
        """
        Unit test to make sure the target is spawned correctly
        """
        # make sure target is not None
        target: StateVector = self.environment.target
        self.assertIsNotNone(target)

        # reset the environnment and see if we still have a target
        self.environment.reset()
        new_target: StateVector = self.environment.target
        self.assertIsNotNone(new_target)
        is_random = self.environment.randomize_target
        if is_random:
            self.assertNotEqual(target, new_target)
        else:
            if target.x != new_target.x:
                self.assertEqual(target.x, new_target.x,
                                 "X values are not equal")
            if target.y != new_target.y:
                self.assertEqual(target.y, new_target.y,
                                 "Y values are not equal")
            if target.z != new_target.z:
                self.assertEqual(target.z, new_target.z,
                                 "Z values are not equal")

    def southwest_corner_mask(self):
        """
        Test that the roll masking is enforced correctly when 
        the agent is near the environment boundaries.

        For this scenario we will:
        - Place the agent near the lower left boundary.
        - The agent is faced north (yaw=0 radians).

        What we want to see is the left roll commands masked with 0 
        when the agent is facing north.

        """
        # Assume that the environment and its configuration are already set up.
        # Get the controlled agent.
        self.agent = self.environment.get_controlled_agents[0]

        # Retrieve the environment boundaries.
        x_min, x_max = self.environment.config.x_bounds
        y_min, y_max = self.environment.config.y_bounds

        # Set the agent's position to be close to a boundary.
        # For example, place the agent near the lower left boundary.
        # 5 units above the lower x-boundary.
        self.agent.state_vector.x = x_min + 200
        # 5 units above the lower y-boundary.
        self.agent.state_vector.y = y_min + 200

        # Set the agent's heading and roll.
        # For instance, set the current yaw to 0 radians (facing east) and roll to 0.
        self.agent.state_vector.yaw_rad = np.deg2rad(0)
        self.agent.state_vector.roll_rad = 0

        # (Optional) Print the agent's state for debugging.
        print("Agent state: x=%.2f, y=%.2f, yaw=%.2f, roll=%.2f" %
              (self.agent.state_vector.x,
               self.agent.state_vector.y,
               self.agent.state_vector.yaw_rad,
               self.agent.state_vector.roll_rad))

        # Get the full action mask (of shape (n_roll, n_alt, n_vel)).
        action_mask = self.environment.get_action_mask()
        print("Full action mask shape:", action_mask.shape)

        # Extract the roll mask from the full action mask.
        # Because the overall mask is created via broadcasting, the roll mask is constant
        # across the altitude and velocity dimensions. For example, we can inspect the first "slice":
        roll_mask = action_mask[:, 0, 0]
        print("Roll mask:", roll_mask)

        # Now compute the desired heading toward the center (as in your get_roll_mask() function).
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        desired_heading = np.arctan2(center_y - self.agent.state_vector.y,
                                     center_x - self.agent.state_vector.x)

        # Compute the heading error.
        heading_error = desired_heading - self.agent.state_vector.yaw_rad
        # Normalize to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        print("Desired heading (radians):", desired_heading)
        print("Heading error (radians):", heading_error)

        # Based on the logic in your roll mask:
        #   - If heading_error > 0, only positive roll deltas should be allowed.
        #   - If heading_error < 0, only negative roll deltas should be allowed.
        #
        # For example, if heading_error > 0, then any roll command that is negative should be masked (set to 0).
        # This should be an array of discrete roll deltas.
        roll_commands = self.environment.roll_commands
        if heading_error > 0:
            # Get indices of negative roll commands.
            neg_indices = np.where(roll_commands < 0)[0]
            for idx in neg_indices:
                self.assertEqual(roll_mask[idx], 0,
                                 msg=f"Roll command at index {idx} (delta {roll_commands[idx]}) "
                                 "should be masked when heading_error > 0.")
        elif heading_error < 0:
            # Get indices of positive roll commands.
            pos_indices = np.where(roll_commands > 0)[0]
            for idx in pos_indices:
                self.assertEqual(roll_mask[idx], 0,
                                 msg=f"Roll command at index {idx} (delta {roll_commands[idx]}) "
                                 "should be masked when heading_error < 0.")
        # else:
        #     # When heading_error is near zero, you might expect only small roll adjustments to be allowed.
        #     for i, roll_delta in enumerate(roll_commands):
        #         if abs(roll_delta) > np.deg2rad(5):
        #             self.assertEqual(roll_mask[i], 0,
        #                              msg=f"Large roll command at index {i} (delta {roll_delta}) "
        #                              "should be masked when heading is nearly aligned.")

    def test_action_masking(self):
        # Set agent's state (modify these values to test different scenarios)
        agent = self.environment.get_controlled_agents[0]
        agent.state_vector.x = self.environment.config.x_bounds[0] + 5
        agent.state_vector.y = self.environment.config.y_bounds[0] + 5
        agent.state_vector.yaw_rad = 0
        agent.state_vector.roll_rad = 0

        # Print the agent's state for verification
        print("Agent state: x =", agent.state_vector.x,
              "y =", agent.state_vector.y,
              "yaw =", agent.state_vector.yaw_rad,
              "roll =", agent.state_vector.roll_rad)

        # Get individual masks (if you have access to them)
        roll_mask = self.environment.get_roll_mask(agent)
        alt_mask = np.ones(len(self.environment.alt_commands), dtype=np.int8)
        vel_mask = np.ones(
            len(self.environment.airspeed_commands), dtype=np.int8)

        print("roll_mask:", roll_mask)
        print("alt_mask:", alt_mask)
        print("vel_mask:", vel_mask)

    def test_env_step(self):
        """

        """
        self.environment.reset()
        agent = self.environment.get_controlled_agents[0]
        n_steps: int = self.environment.config.num_env_steps
        n_roll, n_alt, n_vel = self.environment.action_space.nvec

        for i in range(n_steps):
            # Take a random action.
            action = self.environment.action_space.sample()
            observation, reward, terminated, _, info = self.environment.step(
                action)

            if terminated:
                break

            # print("Observation:", observation["observations"])
            action_mask = observation["action_mask"]
            assert action_mask.shape == (n_roll + n_alt + n_vel,)
            print("Action mask shape:", action_mask.shape)
            print("Action mask:", n_roll, n_alt, n_vel)

            # get the altitude mask
            # print("agent outside", agent.state_vector.array)
            # break the action mask into its components
            roll_mask = action_mask[0:n_roll]
            alt_mask = action_mask[n_roll:n_roll + n_alt]
            assert (len(alt_mask) == len(self.environment.alt_commands))

    def test_rewards(self) -> bool:
        """
        Unit test to make sure the rewards are computed correctly
        """

        # Get the agent
        agent = self.environment.get_controlled_agents[0]
        # Get the target
        target = self.environment.target

        # Get the initial distance to the target
        initial_distance = agent.state_vector.distance_3D(target)

        # Set the agent's position to be close to the target
        agent.state_vector.x = target.x
        agent.state_vector.y = target.y
        agent.state_vector.z = target.z

        # Get the new distance to the target
        new_distance = agent.state_vector.distance_3D(target)

        # Check if the reward is computed correctly
        if new_distance < initial_distance:
            # The agent is closer to the target
            reward = self.environment.compute_intermediate_reward()
            self.assertGreater(reward, 0)
        elif new_distance > initial_distance:
            # The agent is farther from the target
            reward = self.environment.compute_intermediate_reward()
            self.assertLess(reward, 0)
        else:
            # The agent is at the same distance from the target
            reward = self.environment.compute_intermediate_reward()
            self.assertEqual(reward, 0)

        # test to see if terminal reward is computed correctly
        action = self.environment.action_space.sample()
        observation, reward, terminated, _, info = self.environment.step(
            action)
        if terminated:
            print("Agent has reached the target", reward)
            self.assertEqual(reward, self.environment.config.terminal_reward)

        # return True


if __name__ == '__main__':
    unittest.main()
