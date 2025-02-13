import unittest
# from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.simple_multi_env import AbstracKinematicEnv, EngageEnv
from jarvis.utils.trainer import load_yaml_config
from typing import List, Dict, Any


class TestSimpleEnv(unittest.TestCase):

    def setUp(self):
        """
        Test to make sure we are loading our configurations into our environment
        correctly
        """
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.abstract_env = EngageEnv(
            config=self.config)

    def test_init_battlespace(self):
        """
        """

        x_bounds = self.config['bounds']['x']
        y_bounds = self.config['bounds']['y']
        z_bounds = self.config['bounds']['z']

        assert self.abstract_env.battlespace.x_bounds == x_bounds
        assert self.abstract_env.battlespace.y_bounds == y_bounds
        assert self.abstract_env.battlespace.z_bounds == z_bounds

    def test_state_control_limits(self):
        """
        Test to make sure we are are getting the correct state and control limits
        """
        num_evaders = self.config['agents']['num_evaders']
        num_pursuers = self.config['agents']['num_pursuers']

        evader_control_limits, evader_state_limits = self.abstract_env.load_limit_config(
            is_pursuer=False)

        pursuer_control_limits, pursuer_state_limits = self.abstract_env.load_limit_config(
            is_pursuer=True)

        # print("evader_control_limits: ", evader_control_limits)
        # print("evader_state_limits: ", evader_state_limits)

    def test_spawn_agents(self):
        """
        Test to make sure we are spawning our agents correctly
        """
        num_evaders: int = self.config['agents']['num_evaders']
        num_pursuers: int = self.config['agents']['num_pursuers']
        is_pursuer_controlled: bool = self.config['agents']['pursuers']['is_controlled']
        is_evader_controlled: bool = self.config['agents']['evaders']['is_controlled']

        all_agents: List[int] = self.abstract_env.get_all_agents

        assert len(all_agents) == num_evaders + num_pursuers
        print("len(all_agents): ", len(all_agents),
              "num_evaders + num_pursuers: ", num_evaders + num_pursuers)

        controlled_agents: List[int] = self.abstract_env.get_controlled_agents
        print("controlled_agents: ", controlled_agents)

    def test_engageEnv(self):
        """
        Test to make sure we are engaging our environment correctly
        """
        engage_env = EngageEnv(config=self.config)


if __name__ == "__main__":
    unittest.main()
