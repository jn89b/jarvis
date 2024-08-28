from abc import ABC
import numpy as np
import casadi as ca
import copy
from typing import Dict, List, Optional, Text, Tuple, TypeVar

from matplotlib import pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config_2d as env_config

class Obstacle():
    def __init__(self, x:float, y:float, z:float, radius:float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.state_vector = StateVector(x=x, y=y, z=z,
            roll_rad=0, pitch_rad=0, yaw_rad=0, speed=0)

    def is_colliding(self, agent:"Agent") -> bool:
        distance = agent.distance_to(self)
        if distance < self.radius + agent.radius_bubble:
            return True
        return False
    
    

class Agent():
    is_pursuer = None
    def __init__(self,
                 battle_space:BattleSpace,
                 state_vector:StateVector,
                 id:int, 
                 radius_bubble:float,
                 is_controlled:bool=False) -> None:
        self.battle_space = battle_space
        self.plane = Plane2D()
        self.plane.set_state_space()
        
        # the state vector object is x,y,z,roll,pitch,yaw,speed
        # need to transform down to a 4D state vector
        new_array = np.array([
            state_vector.x, state_vector.y, state_vector.yaw_rad, state_vector.speed])
        self.plane.set_info(new_array)
        
        self.state_vector = state_vector
        
        self.id = id
        self.is_controlled = is_controlled
        self.radius_bubble = radius_bubble

        self.action = None
        self.crashed = False

    def __repr__(self) -> Text:
        return f"Agent id {self.id}, {self.state_vector}, controlled: {self.is_controlled}"
    
    @classmethod
    def spawn(self, state_vector:StateVector) -> None:
        print

    def handle_collision(self) -> None:
        agents = self.battle_space.agents
        for agent in agents:
            if agent.id != self.id:
                if self.is_colliding(self.radius_bubble):
                    self.crashed = True
                    agent.crashed = True
                    break
                
                
    def is_colliding(self, distance_threshold:float) -> bool:
        """
        Check if the agent is colliding with another agent
        """
        for agent in self.battle_space.agents:
            if agent.id != self.id:
                distance = self.distance_to(agent)
                if distance < distance_threshold:
                    return True

        return False
    
    def distance_to(self, other:"Agent", 
                    use_2d:bool=False) -> float:
        
        if use_2d:
            return self.state_vector.distance_2D(other.state_vector)
        else:
            return self.state_vector.distance_3D(other.state_vector)
    
    def heading_difference(self, other:"Agent") -> float:        
        return self.state_vector.heading_difference(other.state_vector)

    def is_close_to_parallel(self, other:"Agent", threshold:float=0.7) -> bool:
        dot_product = self.state_vector.dot_product_2D(other.state_vector)
        if dot_product > threshold:
            return True
        
        return False
    
    def act(self, action:np.ndarray) -> None:
        """
        Set the vehicles action
        """
        if action:
            self.action = action
    
    def clip_actions(self) -> np.array:
        """
        Clip the actions to the bounds of the environment
        """
        pass
    
    def on_state_update(self, new_states:np.ndarray) -> None:
        """
        Update the state of the agent
        Note since we are in 2D we only care about 
        x,y,psi,v
        """
        # new_vector = StateVector(
        #     new_states[0], new_states[1], new_states[2],
        #     new_states[3], new_states[4], new_states[5],
        #     new_states[6])
        new_vector = StateVector(
            new_states[0], new_states[1], new_states[2],
            0, 0, new_states[2], new_states[3] 
        )
        x = new_vector.x
        y = new_vector.y
        psi = new_vector.yaw_rad
        v = new_vector.speed
        new_state_info = np.array([x, y, psi, v])
        #we'll keep this information for now
        self.state_vector = new_vector 
        self.plane.set_info(new_states)
    
    def fall_down(self, dt:float) -> None:
        """
        Make the agent fall out of the sky
        """
        velocity = self.state_vector.speed
        new_z = self.state_vector.z - velocity*dt
        self.state_vector.update(z=new_z)
        self.plane.set_info(self.state_vector.array)
    
    def step(self, dt:float) -> None:
        """
        Take a step in the environment
        raise NotImplementedError("Subclass must implement abstract method")
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def get_observation(self) -> np.ndarray:
        """
        Get the observation of the agent
        """
        #ego_state = self.state_vector.array
        # we only want to return the x,y,psi,v
        x = self.state_vector.x
        y = self.state_vector.y
        psi = self.state_vector.yaw_rad
        v = self.state_vector.speed
        
        ego_state = np.array([x, y, psi, v])
        # print("Ego state: ", ego_state)
        
        return ego_state

    
class Evader(Agent):
    is_pursuer = False
    is_controlled = True
    MIN_SPEED_MS = env_config.evader_observation_constraints['airspeed_min']
    MAX_SPEED_MS = env_config.evader_observation_constraints['airspeed_max']
    def __init__(self,
                 battle_space:BattleSpace,
                 state_vector:StateVector,
                 id:int, 
                 radius_bubble:float,
                 is_controlled:bool=True) -> None:
        super().__init__(battle_space, 
                         state_vector, 
                         id, 
                         radius_bubble, 
                         is_controlled)
        self.old_distance_from_pursuer = None 
        self.old_relative_heading = None   
        
    def act(self, action:np.array=None) -> None:
        """
        Set the vehicles action 
        """
        if action.any():
            #TODO: have a check to make sure the action space size is correct
            self.action = action
            
    def step(self, dt:float) -> None:
        """
        Take a step in the environment
        """
        #check if the agent is crashed
        if self.crashed:
            # we want to make the agent fall out of the sky
            # and not take any actions
            self.fall_down(dt)
            return 
        
        self.clip_actions()
        #have to do this since self.action is immutable
        #action_cmd = self.action.copy()
        # action_cmd[2] = -action_cmd[2]
        new_states = self.plane.rk45(
            self.plane.state_info, self.action, dt)
        self.on_state_update(new_states)

    def get_reward(self, obs:np.ndarray) -> float:
        """
        We want to reward the agent for maximizing distance and 
        minimizing the distance between the pursuer
        """
        relative_information = obs[4:]
        #get the first element and every 3rd element
        rel_distances = relative_information[0::3]
        #get the second element and every 3rd element
        rel_velocities = relative_information[1::3]
        #get the third element and every 3rd element
        rel_headings = relative_information[2::3]    

        #choose the worst case scenario for now?
        closest_distance = np.min(rel_distances)
        closest_distance_index = np.argmin(rel_distances)
        closest_velocity = rel_velocities[closest_distance_index]
        
        avg_distance = np.mean(rel_distances)
        avg_velocity = np.mean(rel_velocities)
        avg_heading = np.mean(rel_headings)
        
        #reward the agent for maximizing distance and minimizing the distance between the pursuer
        # reward =  abs(avg_heading)  + 0.1 
        distance_reward = 0.0
        time_reward = 0.1    
        manuever_reward = 0.0
        
        if self.old_distance_from_pursuer is None:
            self.old_distance_from_pursuer = closest_distance
            self.max_distance_from = closest_distance
        else:
            # we want to reward the agent for increasing the distance
            # so if old distance was 5 and new distance is 4 that means the agent is getting closer
            # which is bad, so this will be negative
            distance_reward = closest_distance - self.old_distance_from_pursuer
            self.old_distance_from_pursuer = closest_distance
            # distance_reward = np.clip(distance_reward, -1, 1)
        
        if self.old_relative_heading is None:
            self.old_relative_heading = avg_heading
        else:
            manuever_reward = abs(avg_heading - self.old_relative_heading)
            self.old_relative_heading = avg_heading
            manuever_reward = np.clip(manuever_reward, 0, 0.5)
            
        change_heading = abs(avg_heading)    
        
        alpha = 0.2
        beta = 0.1
        gamma = 0.1

        #reward = change_heading#beta*manuever_reward #+ time_reward + change_heading
        
        other_agents = self.battle_space.agents
        for agent in other_agents:
            if agent.id != self.id and agent.is_pursuer:
                pursuer: Pursuer = agent
                #ideally we want to be faster than the pursuer 
                diff_velocity = self.state_vector.speed - pursuer.state_vector.speed
                
        #we want to minimize the dot product so we add a negative
        # reward = -dot_product + distance_reward
        # reward = -(1 - closest_distance/self.max_distance_from)
        #clip the reward
        
        #reward = alpha*distance_reward #+ beta*manuever_reward
        reward = closest_distance*0.01
        #normalize the distance reward
        
        return reward

class Pursuer(Agent):
    is_pursuer = True
    is_controlled = False    
    MIN_SPEED_MS = env_config.pursuer_observation_constraints['airspeed_min']
    MAX_SPEED_MS = env_config.pursuer_observation_constraints['airspeed_max']
    def __init__(self,
                    battle_space:BattleSpace,
                    state_vector:StateVector,
                    id:int,
                    radius_bubble:float,
                    is_controlled:bool=False,
                    capture_distance:float=0.0,
                    use_pn:bool=True) -> None:
        super().__init__(battle_space, 
                            state_vector, 
                            id, 
                            radius_bubble, 
                            is_controlled)
        self.is_pursuer = True
        self.use_pn = use_pn
        self.capture_distance = capture_distance
        self.observation_constraints = env_config.pursuer_observation_constraints
        self.control_constraints = env_config.pursuer_control_constraints
        if self.use_pn:
            self.pro_nav = ProNav(env_config.DT)
    
    def pro_nav_guidance(self) -> Tuple[float, float]:
        """
        Use ProNav to guide the pursuer to the target
        Returns the yaw rate and acceleration command for the pursuer
        """
        for agent in self.battle_space.agents:
            if agent.id != self.id and agent.is_pursuer == False:
                a_cmd, yaw_rate = self.pro_nav.pursuit(
                    self.state_vector, agent.state_vector)

        return yaw_rate, a_cmd

    def clip_actions(self) -> np.array:
        """
        Clip the actions to the bounds of the environment
        """
        self.action[0] = np.clip(self.action[0], 
                                            self.control_constraints['u_psi_min'], 
                                            self.control_constraints['u_psi_max'])
        self.action[1] = np.clip(self.action[1], 
                                            self.control_constraints['v_cmd_min'], 
                                            self.control_constraints['v_cmd_max'])
    
    def get_observation(self) -> np.ndarray:
        """
        Get the observation of the agent
        """
        ego_state = self.state_vector.array
        return ego_state
    
    def step(self, dt:float) -> None:
        """
        Take a step in the environment
        """
        #check if the agent is crashed
        if self.crashed:
            # we want to make the agent fall out of the sky
            # and not take any actions
            self.fall_down(dt)
            return 
        
        if self.use_pn:
            yaw_rate, a_cmd = self.pro_nav_guidance()
            self.action = np.array([yaw_rate, a_cmd])

        #copy the action
        #self.action = copy(self.action)
        self.action[1] = self.action[1] + self.state_vector.speed
        self.clip_actions()
        
        #set the acceleration 
        new_states = self.plane.rk45(
            self.plane.state_info, self.action, dt)
        self.on_state_update(new_states)
    
class DataHandler():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.yaw = []
        self.u = []
        self.time = []
        self.rewards = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.yaw.append(info_array[2])
        self.u.append(info_array[3])

    def update_reward(self, reward:float) -> None:
        self.rewards.append(reward)
        
    def update_time(self, time:float) -> None:
        self.time.append(time)

class Plane2D():
    def __init__(self, 
                 include_time:bool=False,
                 dt_val:float=env_config.DT,
                 max_roll_dg:float=45,
                 max_pitch_dg:float=25,
                 min_airspeed_ms:float=12,
                 max_airspeed_ms:float=30) -> None:
        self.include_time = include_time
        self.dt_val = dt_val
        self.define_states()
        self.define_controls() 
        
        self.max_roll_rad = np.deg2rad(max_roll_dg)
        self.max_pitch_rad = np.deg2rad(max_pitch_dg)
        self.min_airspeed_ms = min_airspeed_ms
        self.max_airspeed_ms = max_airspeed_ms
        self.airspeed_tau = 0.05 #response of system to airspeed command
        self.pitch_tau = 0.02 #response of system to pitch command
        self.state_info = None
        self.data_handler = DataHandler()
    
    def set_info(self, state_info:np.ndarray) -> None:
        self.state_info = state_info
        self.data_handler.update_data(state_info)
    
    def set_time(self, time:float) -> None:
        self.data_handler.update_time(time)
    
    def get_info(self) -> np.ndarray:
        return self.state_info
    
    def define_states(self):
        """define the states of your system"""
        #positions off the world in NED Frame
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.psi_f = ca.SX.sym('psi_f')
        self.v = ca.SX.sym('v')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.psi_f,
            self.v 
        )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """
        controls for your system
        Controls are turn rate and airspeed
        """
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """
        define the state space of your system
        NED Frame
        """
        
        self.v_dot = (self.v_cmd - self.v)*(self.dt_val/self.airspeed_tau)
        self.x_fdot = self.v * ca.cos(self.psi_f)
        self.y_fdot = self.v * ca.sin(self.psi_f)
        # self.psi_fdot = self.u_psi*self.dt_val #(self.u_psi - self.psi_f)*(self.dt_val/self.pitch_tau)
        self.psi_fdot = self.u_psi#*(self.pitch_tau)
    
        if self.include_time:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.psi_fdot,
                self.v_dot
            )
        else:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.psi_fdot,
                self.v_dot
            )

        #ODE function
        self.function = ca.Function('f', 
            [self.states, self.controls], 
            [self.z_dot])
        
    def update_reward(self, reward:float) -> None:
        self.data_handler.update_reward(reward)
        
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """
        Runge-Kutta 4th order integration
        x is the current state
        u is the current control input
        dt is the time step
        use_numeric is a boolean to return the result as a numpy array
        """
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            #wrap the yaw angle
            next_step[2] = (next_step[2] + np.pi) % (2 * np.pi) - np.pi
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
