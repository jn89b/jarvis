"""
Include radars, agents, terrain, etc.
"""
import numpy as np

from typing import List, Tuple, TYPE_CHECKING, Dict
from jarvis.utils.vector import StateVector
from jarvis.envs.tokens import ControlIndex
# from jarvis.assets.Radar2D import RadarSystem2D
if TYPE_CHECKING:
    # from jarvis.envs.agent import Agent, Pursuer, Evader

    from jarvis.envs.simple_agent import SimpleAgent as Agent
    from jarvis.envs.simple_agent import Pursuer, Evader


class BattleSpace():
    def __init__(self,
                 x_bounds: np.ndarray,
                 y_bounds: np.ndarray,
                 z_bounds: np.ndarray,
                 agents: List["Agent"] = None) -> None:
        self.x_bounds: np.ndarray = x_bounds
        self.y_bounds: np.ndarray = y_bounds
        self.z_bounds: np.ndarray = z_bounds
        self.all_agents: List[Agent] = agents

    def is_out_bounds(self, state_vector: StateVector) -> bool:
        """
        Check if the state vector is out of bounds.
        """
        x, y, z = state_vector.x, state_vector.y, state_vector.z
        return (x < self.x_bounds[0] or x > self.x_bounds[1] or
                y < self.y_bounds[0] or y > self.y_bounds[1] or
                z < self.z_bounds[0] or z > self.z_bounds[1])

    def act(self, action: Dict, use_multi: bool = False) -> None:
        """
        Set the actions on the environment.
        """
        if use_multi:
            for agent in self.all_agents:
                if agent.is_controlled:
                    # check if key is in action
                    if agent.agent_id in action:
                        agent.act(action[agent.agent_id])
                    # else:
                    #     raise ValueError(
                    #         "Action key not found for agent but is supposed to be controlled {}".format(agent.agent_id))
        else:
            controlled_agent: Agent = None
            for agent in self.all_agents:
                if agent.is_controlled:
                    agent.act(action)
                else:
                    agent: Pursuer

                    agent.chase(controlled_agent)
                    # for agent in self.all_agents:
                    #     agent: Agent
                    #     if agent.is_pursuer and not agent.is_controlled:
                    #         agent: Pursuer
                    #         agent.chase(controlled_agent)

    def check_captured(self) -> bool:
        """
        Check if the evader has been captured.
        """
        # get other agents that are not pursuers

        return False

    def check_collisions(self, ego_agent: "Agent", other_agent: "Agent") -> None:
        distance = ego_agent.distance_to(other_agent)
        if ego_agent.is_pursuer:
            threshold = ego_agent.capture_radius + other_agent.radius_bubble
        elif other_agent.is_pursuer:
            threshold = other_agent.capture_radius + ego_agent.radius_bubble
        else:
            threshold = ego_agent.radius_bubble + other_agent.radius_bubble

        if distance <= threshold:
            print("Collision between agents {} and {}, distance {}".format(
                ego_agent.agent_id, other_agent.agent_id, distance))
            ego_agent.crashed = True
            other_agent.crashed = True

    def clear_jsbsim(self) -> None:
        """
        Close the JSBSim simulation.
        """
        for agent in self.all_agents:
            agent.sim_interface.close_sim()

    def clear_agents(self) -> None:
        self.all_agents: List[Agent] = []

    def step(self) -> None:
        """
        Step the environment.
        """
        # make the pursuer act first
        for agent in self.all_agents:
            agent: Agent
            if agent.actions is None:
                continue
            if agent.is_pursuer:
                agent.step()

        for agent in self.all_agents:
            agent: Agent
            if agent.actions is None:
                continue
            if not agent.is_pursuer:
                agent.step()

        # check for collisions
        for agent in self.all_agents:
            agent: Agent

            for other_agent in self.all_agents:
                other_agent: Agent

                # ignore collisions with pursuers for now
                if agent.is_pursuer and other_agent.is_pursuer:
                    continue

                # if agent.id is None or other_agent.id is None:
                #     if agent.agent_id == other_agent.agent_id:
                #         continue
                if agent.agent_id != other_agent.agent_id:
                    self.check_collisions(agent, other_agent)
                # if agent.id != other_agent.id:
                #     self.check_collisions(agent, other_agent)

        # check out of bounds
        for agent in self.all_agents:
            if self.is_out_bounds(agent.state_vector):
                agent.crashed = True

    def step_single_agent(self, agent: "Agent") -> None:
        """
        Step a single agent.
        """
        if agent.actions is None:
            return

        agent.step()

        # check for collisions
        for agent in self.all_agents:
            agent: Agent

            for other_agent in self.all_agents:
                other_agent: Agent

                # ignore collisions with pursuers for now
                if agent.is_pursuer and other_agent.is_pursuer:
                    continue

                # if agent.id is None or other_agent.id is None:
                #     if agent.agent_id == other_agent.agent_id:
                #         continue
                if agent.agent_id != other_agent.agent_id:
                    self.check_collisions(agent, other_agent)
                # if agent.id != other_agent.id:
                #     self.check_collisions(agent, other_agent)

        # check out of bounds
        for agent in self.all_agents:
            if self.is_out_bounds(agent.state_vector):
                agent.crashed = True
