import unittest
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv
# from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)

from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.vector import StateVector
from jarvis.envs.battlespace import BattleSpace
from typing import List, Dict, Any


plt.close('all')


class TestHRLEnv(unittest.TestCase):
    def setUp(self):
        env_config = load_yaml_config(
            "config/simple_env_config.yaml")['battlespace_environment']
        self.env = HRLMultiAgentEnv(config=env_config)
        # check instance
        if isinstance(self.env, MultiAgentEnv):
            print("MultiAgentEnv")
        else:
            raise ValueError("Not a MultiAgentEnv")
        # check if
        self.assertTrue(isinstance(self.env, HRLMultiAgentEnv))

    def test_observation_space(self) -> None:
        """
        Idiot check to make sure the observation space
        for my good guy consist of:
            - 1 + n_pursuers
                - 1 is the euclidean distance from the good guy to the target
                - n_pursuers is the euclidean distance from the good guy to the pursuers

        """
        n_pursuers: int = len(self.env.get_pursuer_agents())
        n_evaders: int = self.env.get_evader_agents()
        total_obs = 1 + n_pursuers
        # good_guy_obs: gym.spaces.Box = self.env.good_guy_obs_space()
        good_guy_obs = self.env.observation_spaces['good_guy_hrl']['observations']
        print("good guy obs", good_guy_obs.shape)
        print("total obs", total_obs)
        assert good_guy_obs.shape[0] == total_obs

    def test_action_space(self) -> None:
        """
        Idiot check to make sure we have the correct action space 
        with the good guy
        """
        action_space = self.env.action_spaces
        good_guy_key: str = self.env.good_guy_hrl_key
        offensive_key: str = self.env.good_guy_offensive_key
        defensive_key: str = self.env.good_guy_defensive_key

        assert good_guy_key in action_space
        assert offensive_key in action_space
        assert defensive_key in action_space

        # make sure the good guy action space is size 2
        print("good guy action space", action_space[good_guy_key])

    # def test_agents(self) -> None:
    #     all_agents: List[SimpleAgent] = self.env.get_all_agents
    #     assert "good_guy" in [agent.agent_id for agent in all_agents]
    #     controlled_agents: List[SimpleAgent] = self.env.get_controlled_agents
    #     assert "good_guy" in [agent.agent_id for agent in controlled_agents]

    #     agents: List[str] = self.env.agents
    #     assert "good_guy_hrl" in agents

    def test_steps(self) -> None:
        """

        """
        print("agents", self.env.agents)
        print("actions", self.env.action_spaces)
        # print("observatio     n", self.env.observation_spaces.keys())
        assert self.env.target is not None
        self.env.reset()
        print("agents now ", self.env.agents)
        self.env.current_agent = "1"
        for i in range(1000):
            # agent = next(self.env.agent_cycle)
            # print("agent", agent)
            actions = {agent_id: action_space['action'].sample()
                       for agent_id,
                       action_space in self.env.action_spaces.items()}
            current_action = actions[self.env.current_agent]
            current_action = {
                "action": current_action
            }
            if self.env.current_agent == "good_guy_hrl":
                current_action["action"] = 0
            action_dict = {self.env.current_agent: current_action}
            obs, reward, done, _, info = self.env.step(
                action_dict=action_dict)
            #print("obs", obs.keys())
            # check if obs is empty dictionary
            # if done:
            #     break
            # if not obs:
            #     raise ValueError("obs is empty current agent is",
            #                      self.env.current_agent)
                # print("obs is empty")

        print("agents now ", self.env.agents)
        
        # plot the distance to the target
        # get the good guy

        # plot a 3D plot
        datas: List[DataHandler] = []
        agents = self.env.get_all_agents
        # agents
        new_agents = []
        new_agents.append(self.env.get_evader_agents()[0])
        new_agents.extend(self.env.get_pursuer_agents())
        agents = new_agents
        for agent in agents:
            data: DataHandler = agent.simple_model.data_handler
            datas.append(data)

        # plot a 3D plot of the agents
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, data in enumerate(datas):
            ax.scatter(data.x[0], data.y[1], data.z[2], label=f"Agent Start {i}")
            ax.plot(data.x, data.y, data.z, label=f"Agent {i}")

        target: StateVector = self.env.target
        # plot the goal target as a cylinder
        ax.scatter(target.x, target.y, target.z,
                label="Target", color='red', s=100)

        # print("env step", self.env.current_step)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        # tight axis
        fig.tight_layout()
        ax.legend()
        plt.show()
        

if __name__ == "__main__":
    unittest.main()
