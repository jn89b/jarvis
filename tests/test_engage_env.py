import unittest
import numpy as np
import matplotlib.pyplot as plt

# from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.simple_multi_env import AbstracKinematicEnv, EngageEnv
from jarvis.envs.simple_agent import SimpleAgent, PlaneKinematicModel, DataHandler
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.vector import StateVector
from jarvis.envs.battlespace import BattleSpace
from typing import List, Dict, Any

plt.close('all')


class TestSimpleEnv(unittest.TestCase):

    def setUp(self):
        """
        Test to make sure we are loading our configurations into our environment
        correctly
        """
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.engage_env = EngageEnv(
            config=self.config)
        self.init_agent()

    def init_agent(self) -> None:
        """
        Simple test to spawn an agent and see
        that it spawns to correct location
        """
        x = -975
        y = 50
        z = 50
        roll_rad = 0.1
        pitch_rad = 0.1
        yaw_rad = 0.1
        speed = 25
        self.battlespace = BattleSpace(x_bounds=[0, 100],
                                       y_bounds=[0, 100],
                                       z_bounds=[0, 100])
        state_vector = StateVector(
            x=x,
            y=y,
            z=z,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
            speed=speed
        )
        self.plane_model = PlaneKinematicModel()
        self.agent = SimpleAgent(
            battle_space=self.battlespace,
            state_vector=state_vector,
            agent_id=0,
            simple_model=self.plane_model)

    def test_init_battlespace(self):
        """
        """

        x_bounds = self.config['bounds']['x']
        y_bounds = self.config['bounds']['y']
        z_bounds = self.config['bounds']['z']

        assert self.engage_env.battlespace.x_bounds == x_bounds
        assert self.engage_env.battlespace.y_bounds == y_bounds
        assert self.engage_env.battlespace.z_bounds == z_bounds

    def test_state_control_limits(self):
        """
        Test to make sure we are are getting the correct state and control limits
        """
        num_evaders = self.config['agents']['num_evaders']
        num_pursuers = self.config['agents']['num_pursuers']

        evader_control_limits, evader_state_limits = self.engage_env.load_limit_config(
            is_pursuer=False)

        pursuer_control_limits, pursuer_state_limits = self.engage_env.load_limit_config(
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

        all_agents: List[int] = self.engage_env.get_all_agents

        assert len(all_agents) == num_evaders + num_pursuers
        print("Number of agents: ", len(all_agents),
              "num_evaders + num_pursuers: ", num_evaders + num_pursuers)

        controlled_agents: List[int] = self.engage_env.get_controlled_agents
        print("controlled_agents: ", controlled_agents)

    # def test_pitch_action_masking(self) -> None:
    #     """
    #     Test to make sure we are engaging our environment correctly
    #     """

    #     # test masking
    #     # See if we are clipping the action space for higher pitches
    #     # if we are close
    #     self.agent.battle_space = self.abstract_env.battlespace
    #     max_z: float = self.abstract_env.battlespace.z_bounds[1]
    #     self.agent.state_vector.z = max_z - 3
    #     action_mask: np.array = self.abstract_env.get_action_mask(
    #         self.agent)
    #     assert action_mask.shape == self.abstract_env.action_space.nvec.sum()
    #     unwrapped_mask = self.abstract_env.unwrap_action_mask(action_mask)
    #     print("pitch mask", unwrapped_mask["pitch"])
    #     assert unwrapped_mask["pitch"][-1] == 0

    #     # see if we are clipping the action space for lower pitches
    #     # if we are close
    #     min_z: float = self.abstract_env.battlespace.z_bounds[0]
    #     action_mask: np.array = self.abstract_env.get_action_mask(
    #         self.agent)
    #     self.agent.state_vector.z = min_z + 3
    #     action_mask: np.array = self.abstract_env.get_action_mask(
    #         self.agent)
    #     assert action_mask.shape == self.abstract_env.action_space.nvec.sum()
    #     unwrapped_mask = self.abstract_env.unwrap_action_mask(action_mask)
    #     assert unwrapped_mask["pitch"][0] == 0

    def test_lateral_action_masking(self) -> None:
        """
        Test to make sure we are engaging our environment correctly
        """

        # test masking
        # See if we are clipping the action space for higher pitches
        # if we are close
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.engage_env = EngageEnv(
            config=self.config)
        self.init_agent()
        self.agent = self.engage_env.get_controlled_agents[0]
        max_x: float = self.engage_env.battlespace.x_bounds[1]
        self.agent.state_vector.x = max_x - 3
        print("agent state vector", self.agent.state_vector)
        action_mask: np.array = self.engage_env.get_action_mask(
            self.agent)
        assert action_mask.shape == self.engage_env.action_space.nvec.sum()
        unwrapped_mask = self.engage_env.unwrap_action_mask(action_mask)
        print("lateral mask", unwrapped_mask["yaw"])
        # assert unwrapped_mask["yaw"][-1] == 0

        # see if we are clipping the action space for lower pitches
        # if we are close
        min_x: float = self.engage_env.battlespace.x_bounds[0]
        action_mask: np.array = self.engage_env.get_action_mask(
            self.agent)
        self.agent.state_vector.x = min_x + 3
        action_mask: np.array = self.engage_env.get_action_mask(
            self.agent)
        assert action_mask.shape == self.engage_env.action_space.nvec.sum()
        unwrapped_mask = self.engage_env.unwrap_action_mask(action_mask)
        print("lateral mask", unwrapped_mask["yaw"])

    def test_pitch_mask_target_above(self) -> None:
        """
        A test to make sure we are masking the pitch action space
        based on the target location, we want to negative 
        pitches are masked if the target is above us
        and positive pitches are masked if the target is below us
        """
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.engage_env = EngageEnv(
            config=self.config)
        agent = self.engage_env.get_controlled_agents[0]

        # the target is above us so let's mask the negative pitches
        agent.state_vector.z = 50
        self.engage_env.target.z = 75

        observation = self.engage_env.observe(agent)
        action_mask = observation["action_mask"]

        # unwrap the mask
        mask_unwrapped = self.engage_env.unwrap_action_mask(action_mask)
        pitch_mask = mask_unwrapped["pitch"]
        print("pitch mask", pitch_mask)
        assert pitch_mask[0] == 0

        self.engage_env.target.z = 35
        observation = self.engage_env.observe(agent)
        action_mask = observation["action_mask"]
        # unwrap the mask
        mask_unwrapped = self.engage_env.unwrap_action_mask(action_mask)
        pitch_mask = mask_unwrapped["pitch"]
        print("pitch mask", pitch_mask)
        assert pitch_mask[-1] == 0

    def test_yaw_mask_target_right(self) -> None:
        """
        Similar to the pitch masking test, we want to mask
        the yaw action space based on the target location
        """
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.engage_env = EngageEnv(
            config=self.config)
        agent = self.engage_env.get_controlled_agents[0]
        agent.state_vector.x = 0
        agent.state_vector.y = 0
        agent.state_vector.yaw_rad = np.deg2rad(0)

        # the target is to the left of us so let's mask the negative yaws
        self.engage_env.target.x = 0
        self.engage_env.target.y = 100

        observation = self.engage_env.observe(agent)
        action_mask = observation["action_mask"]
        mask_unwrapped = self.engage_env.unwrap_action_mask(action_mask)
        yaw_mask = mask_unwrapped["yaw"]
        print("yaw mask", yaw_mask)

    def test_intermediate_reward(self) -> None:
        """
        Test to make sure we the closer we are to 
        our target then we get a higher reward
        """
        # Get a reference to the agent once
        self.config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.engage_env = EngageEnv(
            config=self.config)
        agent = self.engage_env.get_controlled_agents[0]

        # Update its attributes
        agent.state_vector.x = 0
        agent.state_vector.y = 0
        agent.state_vector.z = 50
        agent.state_vector.yaw_rad = np.deg2rad(10)
        agent.simple_model.set_state_space(agent.state_vector)

        # self.agent.state_vector.target_x = 100
        # self.agent.state_vector.target_y = 100
        # self.agent.state_vector.target_z = 100
        target: StateVector = self.engage_env.target

        n_steps: int = 150
        reward_history: List[float] = []
        info_hisory: List[Dict] = []
        for _ in range(n_steps):
            # just go forward
            # action = np.array([0, 0, 0, 3])
            # get a random action
            action = self.engage_env.action_space.sample()

            observation, reward, terminated, truncated, info = self.engage_env.step(
                action)
            info_hisory.append(info)
            reward_history.append(reward)
            if terminated:
                break

        agent = self.engage_env.get_controlled_agents[0]

        print("agent state vector", agent.state_vector)

        data: DataHandler = agent.return_data()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot(data.x, data.y, data.z)
        ax.scatter(data.x[0], data.y[0], data.z[0], c='r', label='start')
        ax.scatter(target.x, target.y, target.z, c='g', label='target')
        ax.set_xlim(-200, 300)
        ax.set_ylim(-200, 300)

        # set x and y limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        fig, ax = plt.subplots()
        ax.plot(reward_history)
        ax.set_title("Intermediate Reward")

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(np.rad2deg(data.phi), label='phi')
        ax[0].plot(np.rad2deg(data.u_phi), c='r', label='phi_cmd')
        ax[1].plot(np.rad2deg(data.theta), label='theta')
        ax[1].plot(np.rad2deg(data.u_theta), c='r', label='theta_cmd')
        ax[2].plot(np.rad2deg(data.psi))
        ax[2].plot(np.rad2deg(data.u_psi), c='r', label='psi_cmd')
        ax[3].plot(data.v, label='velocity')
        ax[3].plot(data.v_cmd, c='r', label='velocity_cmd')

        for a in ax:
            a.legend()

        plt.show()


if __name__ == "__main__":
    unittest.main()
