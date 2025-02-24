import unittest
import numpy as np
import matplotlib.pyplot as plt

from ray.rllib.env import MultiAgentEnv
# from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.simple_multi_env import PursuerEvaderEnv
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)

from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.vector import StateVector
from jarvis.envs.battlespace import BattleSpace
from typing import List, Dict, Any


plt.close('all')


class TestMultiEnv(unittest.TestCase):

    def setUp(self) -> None:
        env_config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.env = PursuerEvaderEnv(config=env_config)
        # check instance
        if isinstance(self.env, MultiAgentEnv):
            print("MultiAgentEnv")
        else:
            raise ValueError("Not a MultiAgentEnv")
        # check if
        self.assertTrue(isinstance(self.env, PursuerEvaderEnv))

    def test_agents(self) -> None:
        print("control agents", self.env.get_controlled_agents)
        observation_spaces = self.env.observation_spaces
        action_spaces = self.env.action_spaces
        print("action_spaces", action_spaces)
        for k, v in observation_spaces.items():
            observations = v['observations']
            print("observations", observations.shape)
            print("observations low ", observations.low)
            print("observations high", observations.high)
        assert isinstance(observation_spaces, Dict)

    def test_steps(self) -> None:
        print("action space", self.env.action_spaces['0'])
        for i in range(10):
            actions = {agent_id: action_space.sample()
                       for agent_id, action_space in self.env.action_spaces.items()}
            print("actions", actions)
            obs, reward, done, _, info = self.env.step(actions)
            # print("obs", obs)
            # print("reward", reward)
            # print("done", done)
            # print("info", info)
            if done['__all__']:
                print("done", done)
                self.env.reset()

    def test_remove_all_agents(self) -> None:
        """
        Load up the environment and 
        clear the agents 
        then add a pursuer agent 
        and an evader agent
        """
        env_config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        env = PursuerEvaderEnv(config=env_config)
        env.remove_all_agents()
        assert len(env.get_pursuer_agents()) == 0
        # insert an agent
        state_vector = StateVector(
            x=0, y=0, z=0, yaw_rad=0, roll_rad=0,
            pitch_rad=0, speed=20)
        pursuer: Pursuer = Pursuer(
            agent_id="0",
            state_vector=state_vector,
            battle_space=env.battlespace,
            simple_model=PlaneKinematicModel(),
            is_controlled=True,
            radius_bubble=5
        )
        env.insert_agent(pursuer)
        assert len(env.get_controlled_agents) == 1

    def test_intermediate_reward(self) -> None:
        """
        Test to make sure we the closer we are to 
        our target then we get a higher reward
        """
        pursuer: SimpleAgent = self.env.get_pursuer_agents()[0]
        evader: SimpleAgent = self.env.get_evader_agents()[0]

        n_steps = 600
        evader_rewards = []
        pursuer_rewards = []
        distance_history = []
        heading_history = []
        dt: float = 0.1
        print("purser", pursuer.state_vector)
        print("evader", evader.state_vector)
        pursuer_history = []
        evader_history = []
        pursuer.state_vector.x = 100
        pursuer.state_vector.y = 100
        pursuer.state_vector.z = 30
        pursuer.state_vector.yaw_rad = evader.state_vector.yaw_rad
        # desired heading
        dx = evader.state_vector.x - pursuer.state_vector.x
        dy = evader.state_vector.y - pursuer.state_vector.y
        desired_heading = np.arctan2(dy, dx)
        for i in range(n_steps):
            # move the pursuer closer to the evader
            pursuer_pos = pursuer.state_vector
            evader_pos = evader.state_vector
            delta_x = evader_pos.x - pursuer_pos.x
            delta_y = evader_pos.y - pursuer_pos.y
            delta_z = evader_pos.z - pursuer_pos.z
            pursuer_pos.x += delta_x / n_steps
            pursuer_pos.y += delta_y / n_steps
            pursuer_pos.z += delta_z / n_steps
            pursuer.state_vector.yaw_rad += np.deg2rad(1)
            reward = self.env.compute_pursuer_reward(pursuer=pursuer,
                                                     evader=evader)
            evader_reward = self.env.compute_evader_reward(pursuer=pursuer,
                                                           evader=evader)
            evader_rewards.append(evader_reward)
            distance_from_evader = np.linalg.norm(
                pursuer.state_vector.array[0:3] - evader.state_vector.array[0:3])
            heading_error = pursuer.state_vector.yaw_rad - \
                evader.state_vector.yaw_rad
            # wrap between -pi and pi
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            heading_command = pursuer.state_vector.yaw_rad + heading_error
            # wrap between -pi and pi
            heading_command = (heading_command + np.pi) % (2 * np.pi) - np.pi
            # pursuer.state_vector.yaw_rad = heading_command

            distance_history.append(distance_from_evader)
            pursuer_rewards.append(reward)
            heading_history.append(np.rad2deg(heading_error))
            print("purser", pursuer.state_vector)
            pursuer_history.append(pursuer.state_vector.array[0:3])
            evader_history.append(evader.state_vector.array[0:3])

        fig, ax = plt.subplots(nrows=3)
        ax[0].plot(pursuer_rewards, label="Pursuer Reward")
        ax[0].plot(evader_rewards, label="Evader Reward")
        ax[1].plot(distance_history, label="Distance")
        ax[2].plot(heading_history, label="Heading Error")
        # horizontal line for desired heading
        ax[2].axhline(np.rad2deg(desired_heading), color='r',
                      linestyle='--', label="Desired Heading")
        for a in ax:
            a.legend()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        pursuer_history = np.array(pursuer_history)
        evader_history = np.array(evader_history)

        ax.plot(pursuer_history[:, 0], pursuer_history[:,
                1], pursuer_history[:, 2], label="Pursuer")
        ax.plot(evader_history[:, 0], evader_history[:, 1],
                evader_history[:, 2], label="Evader")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        plt.show()


if __name__ == '__main__':
    unittest.main()
