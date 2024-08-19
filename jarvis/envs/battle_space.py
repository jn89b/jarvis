"""
Include radars, agents, terrain, etc.
"""
import numpy as np

from typing import List, Tuple, TYPE_CHECKING,Dict
from jarvis.utils.Vector import StateVector
if TYPE_CHECKING:
    from jarvis.assets.Plane import Agent

class BattleSpace():
    def __init__(self,
                 x_bounds:np.ndarray,
                 y_bounds:np.ndarray,
                 z_bounds:np.ndarray,
                 agents:List["Agent"]=None) -> None:
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.agents = agents
        
    def is_out_bounds(self, state_vector:StateVector) -> bool:
        """
        Check if the state vector is out of bounds.
        """
        x, y, z = state_vector.x, state_vector.y, state_vector.z
        return (x < self.x_bounds[0] or x > self.x_bounds[1] or
                y < self.y_bounds[0] or y > self.y_bounds[1] or
                z < self.z_bounds[0] or z > self.z_bounds[1])
        
    def act(self, action:Dict, use_multi:bool=True) -> None:
        """
        Act on the environment.
        """
        if use_multi:
            for agent in self.agents:
                if agent.is_controlled:
                    #check if key is in action
                    if agent.id in action:
                        agent.act(action[agent.id])
                    else:
                        raise ValueError("Action key not found for agent but is supposed to be controlled {}".format(agent.id))
        else:
            for agent in self.agents:
                if agent.is_controlled:
                    agent.act(action)


    def check_captured(self) -> bool:
        """
        Check if the evader has been captured.
        """
        #get other agents that are not pursuers
        
        return False

    def check_collisions(self, ego_agent:"Agent", other_agent:"Agent") -> None:
        distance = ego_agent.distance_to(other_agent)
        if ego_agent.is_pursuer:
            threshold = ego_agent.capture_distance + other_agent.radius_bubble
        elif other_agent.is_pursuer:
            threshold = other_agent.capture_distance + ego_agent.radius_bubble
        else:
            threshold = ego_agent.radius_bubble + other_agent.radius_bubble
            
        if distance <= threshold:
            ego_agent.crashed = True
            other_agent.crashed = True
            # print("Collision between agent {} and agent {}".format(
            #     ego_agent.id, other_agent.id))

    def step(self, dt:float) -> None:
        """
        Step the environment.
        """
        # for agent in self.agents:
        #     agent.step(dt)
        
        #make the pursuer act first
        for agent in self.agents:
            agent: Agent
            if agent.is_pursuer:
                agent.step(dt)
        
        for agent in self.agents:
            agent: Agent
            if not agent.is_pursuer:
                agent.step(dt)    
        
        #check for collisions
        for agent in self.agents:
            agent: Agent
    
            for other_agent in self.agents:
                other_agent: Agent
                
                #ignore collisions with pursuers for now
                if agent.is_pursuer and other_agent.is_pursuer:
                    continue
                
                if agent.id != other_agent.id:
                    self.check_collisions(agent, other_agent)
        
        #check out of bounds
        for agent in self.agents:
            if self.is_out_bounds(agent.state_vector):
                agent.crashed = True
                # print("Agent {} out of bounds".format(agent.id))