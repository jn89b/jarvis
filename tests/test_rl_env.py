
from jarvis.envs.env import DynamicThreatAvoidance, EnvConfig, AircraftConfig
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.agent import Agent
from aircraftsim import SimInterface
from aircraftsim.utils.report_diagrams import SimResults
from gymnasium import spaces
from typing import List
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import unittest
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('TkAgg')
USE_IMPORT = True

# if USE_IMPORT:
#     sys.path.append('../jarvis')
#     # print the path
#     print(sys.path)


class TestDynamicThreatAvoidance(unittest.TestCase):

    def setUp(self):
        """Set up the environment and configuration for testing."""
        # self.env_config = EnvConfig(
        #     x_bounds=[0, 100],
        #     y_bounds=[0, 100],
        #     z_bounds=[0, 100],
        #     num_evaders=2,
        #     num_pursuers=1,
        #     dt=0.1,
        #     sim_frequency=50,
        #     sim_end_time=10.0
        # )
        # self.aircraft_config = AircraftConfig(
        #     control_limits={
        #         "velocity": {"min": -2.0, "max": 2.0},
        #         "altitude": {"min": -5.0, "max": 5.0},
        #         "roll": {"min": -1.0, "max": 1.0}
        #     },
        #     state_limits={
        #         "x": {"min": -50.0, "max": 50.0},
        #         "y": {"min": -50.0, "max": 50.0},
        #         "z": {"min": 0.0, "max": 50.0},
        #         "v": {"min": 0.0, "max": 30.0}
        #     }
        # )

        self.env = DynamicThreatAvoidance(
            config_file_dir='config.yaml',
            aircraft_config_dir='aircraft_config.yaml'
        )
        self.battlespace: BattleSpace = self.env.battlespace
        self.env_config: EnvConfig = self.env.config
        self.aircraft_config: AircraftConfig = self.env.aircraft_config

    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertIsInstance(self.env.battlespace, BattleSpace)
        self.assertEqual(len(self.env.all_agents),
                         self.env_config.num_evaders + self.env_config.num_pursuers)

    def test_observation_space(self):
        """Test observation space shape and bounds."""
        obs_space = self.env.observation_space
        self.assertIsInstance(obs_space, spaces.Box)
        self.assertEqual(len(obs_space.low), obs_space.shape[0])
        self.assertEqual(len(obs_space.high), obs_space.shape[0])

        # Check bounds
        self.assertTrue(np.all(obs_space.low < obs_space.high))

    def test_action_space(self):
        """Test action space correctness."""
        action_space = self.env.action_space
        self.assertIsInstance(action_space, spaces.Space)

        if isinstance(action_space, spaces.MultiDiscrete):
            self.assertGreaterEqual(np.min(action_space.nvec), 1)

    def test_reset(self):
        """Test environment reset and returned observation."""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(len(info), len(self.env.all_agents))
        self.assertEqual(obs.shape[0], self.env.observation_space.shape[0])

    def test_step_function(self):
        """Test environment step function behavior."""
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(next_obs, np.ndarray)
        # Ensure obs is within bounds
        self.assertTrue(np.all(next_obs >= self.env.observation_space.low))
        self.assertTrue(np.all(next_obs <= self.env.observation_space.high))

    def test_agent_spawning(self):
        """Test agents are spawned within boundaries."""
        for agent in self.env.all_agents:
            pos = agent.state_vector
            self.assertGreaterEqual(pos.x, self.env_config.x_bounds[0])
            self.assertLessEqual(pos.x, self.env_config.x_bounds[1])
            self.assertGreaterEqual(pos.y, self.env_config.y_bounds[0])
            self.assertLessEqual(pos.y, self.env_config.y_bounds[1])
            self.assertGreaterEqual(pos.z, self.env_config.z_bounds[0])
            self.assertLessEqual(pos.z, self.env_config.z_bounds[1])

    def test_agent_control(self):
        """Test agent control limits."""
        num_actions: int = self.env.action_space
        action_0: int = len(self.env.roll_commands)
        action_1: int = len(self.env.altitude_commands)
        action_2: int = len(self.env.velocity_commands)
        action = np.array([0, 0, 0])
        self.env.step(action)

        N_steps: int = 2500
        for i in range(N_steps):
            action = np.array([0, 0, 0])
            obs, reward, done, _, info = self.env.step(action)
            if done:
                break

        # plot a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        agents: Agent = self.env.all_agents

        for i, agent in enumerate(agents):
            report: SimResults = agent.sim_interface.report
            if i == 0:
                label: str = 'Evader'
            else:
                label: str = 'Pursuer' + str(i)
            ax.scatter(report.x[0], report.y[0], report.z[0], label=label)
            ax.plot(report.x, report.y, report.z, label=label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(report.roll_dg, label='Roll')
        ax[0].set_ylabel('Roll (deg)')
        ax[0].legend()
        ax[1].plot(report.pitch_dg, label='Pitch')
        ax[1].set_ylabel('Pitch (deg)')
        ax[1].legend()
        ax[2].plot(report.yaw_dg, label='Yaw')
        ax[2].set_ylabel('Yaw (deg)')
        plt.show()


if __name__ == '__main__':
    unittest.main()
