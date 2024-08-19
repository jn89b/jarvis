from abc import ABC
import numpy as np
import casadi as ca
from typing import Dict, List, Optional, Text, Tuple, TypeVar

from matplotlib import pyplot as plt
from jarvis.utils.Vector import StateVector
from jarvis.envs.battle_space import BattleSpace
from jarvis.algos.pronav import ProNav
from jarvis.config import env_config

class Agent():
    is_pursuer = None
    def __init__(self,
                 battle_space:BattleSpace,
                 state_vector:StateVector,
                 id:int, 
                 radius_bubble:float,
                 is_controlled:bool=False) -> None:
        self.battle_space = battle_space
        self.plane = Plane()
        self.plane.set_state_space()
        self.plane.set_info(state_vector.array)
        
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
    
    def act(self, action:Dict) -> None:
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
        """
        new_vector = StateVector(
            new_states[0], new_states[1], new_states[2],
            new_states[3], new_states[4], new_states[5],
            new_states[6])
        self.state_vector = new_vector
        self.plane.set_info(new_vector.array)
    
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
        """
        #check if the agent is crashed
        if self.crashed:
            # we want to make the agent fall out of the sky
            # and not take any actions
            self.fall_down(dt)
            return 
        
        self.clip_actions()
        yaw_input = self.action['yaw_cmd']
        speed_cmd = self.action['speed_cmd']
        #set the acceleration 
        # speed_input = self.plane.state_info[6] + (acceleration * dt)
        #make sure the speed is within the limits
        # speed_input = np.clip(speed_input, 
        #                       self.MIN_SPEED_MS, 
        #                       self.MAX_SPEED_MS)
        yaw_cmd = yaw_input
        action = np.array([self.action['roll_cmd'],
                            self.action['pitch_cmd'],
                            yaw_cmd,
                            speed_cmd])
        new_states = self.plane.rk45(
            self.plane.state_info, action, dt)
        self.on_state_update(new_states)
        
    def get_observation(self) -> np.ndarray:
        """
        Get the observation of the agent
        """
        ego_state = self.state_vector.array
        #clip the values of the angles
        
        return ego_state
        
class Pursuer(Agent):
    is_pursuer = True
    MIN_SPEED_MS = env_config.pursuer_observation_constraints['airspeed_min']
    MAX_SPEED_MS = env_config.pursuer_observation_constraints['airspeed_max']
    MIN_ROLL = env_config.pursuer_observation_constraints['phi_min']
    MAX_ROLL = env_config.pursuer_observation_constraints['phi_max']
    MIN_PITCH = env_config.pursuer_observation_constraints['theta_min']
    MAX_PITCH = env_config.pursuer_observation_constraints['theta_max']
    def __init__(self,
                 battle_space:BattleSpace,
                 state_vector:StateVector,
                 id:int, 
                 radius_bubble:float,
                 is_controlled:bool=False,
                 capture_distance = 0.0,
                 use_pn:bool=True) -> None:
        super().__init__(battle_space, 
                         state_vector, 
                         id, 
                         radius_bubble, 
                         is_controlled)
        
        self.use_pn = use_pn
        self.capture_distance = capture_distance
        self.old_distance_from_evader = None
        if self.use_pn:
            self.pronav = ProNav(env_config.DT)
    
    def pro_nav_guidance(self) -> Tuple[float, float,
                                        float, float]:
        """
        Use the pronav algorithm to navigate the agent
        """
        for agent in self.battle_space.agents:
            if agent.id != self.id and agent.is_pursuer == False:
                a_cmd, latax = self.pronav.navigate(
                    self.state_vector, agent.state_vector)
                
                pitch_cmd = self.pitch_command(
                    self.state_vector, agent.state_vector)
                
        yaw_rate = latax
        roll_cmd = np.arctan2(yaw_rate, 9.81)
        # roll_cmd = 0

        #have to add negatives to make this work
        # a_cmd = 1
        return roll_cmd, pitch_cmd, yaw_rate, a_cmd
    
    def pitch_command(self, ego:StateVector, target:
        StateVector) -> float:
        distance_two_d = ego.distance_2D(target)
        dz = target.z - ego.z
        
        return np.arctan2(dz, distance_two_d)
    
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

        if self.action is None:
            self.action = {
                'roll_cmd':0,
                'pitch_cmd':0,
                'yaw_cmd':0,
                'speed_cmd':0
            }

        if self.use_pn:
            #TODO: This will be a Policy 
            roll_cmd, pitch_cmd, yaw_rate, vel_add = self.pro_nav_guidance()
            # yaw_input = yaw_rate
            # speed_cmd = vel_additional + self.state_vector.speed
            self.action['yaw_cmd'] = yaw_rate 
            self.action['speed_cmd'] = vel_add + self.state_vector.speed
            self.action['roll_cmd'] = roll_cmd
            self.action['pitch_cmd'] = pitch_cmd
            
        self.clip_actions()
        yaw_cmd = self.action['yaw_cmd']
        speed_cmd = self.action['speed_cmd']
        roll_cmd = np.clip(self.action['roll_cmd'],
                            self.MIN_ROLL, 
                            self.MAX_ROLL)
        pitch_cmd = np.clip(self.action['pitch_cmd'],
                            self.MIN_PITCH, 
                            self.MAX_PITCH)
        #set the acceleration 
        # speed_input = self.plane.state_info[6] + (acceleration * dt)
        #make sure the speed is within the limits
        speed_input = np.clip(speed_cmd, 
                              self.MIN_SPEED_MS, 
                              self.MAX_SPEED_MS)
        # positive pitch command makes it go down
        action = np.array([roll_cmd,
                            -pitch_cmd,
                            yaw_cmd,
                            speed_input])
        
        new_states = self.plane.rk45(
            self.plane.state_info, action, dt)
        self.on_state_update(new_states)
    
    def act(self, action:Dict) -> None:
        """
        Set the vehicles action
        """
        if action:
            self.action = action
        
class Evader(Agent):
    is_pursuer = False
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
            self.action = action
            
    def get_reward(self, obs:np.ndarray) -> float:
        """
        We want to reward the agent for maximizing distance and 
        minimizing the distance between the pursuer
        """
        relative_information = obs[7:]
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
            manuever_reward = np.clip(manuever_reward, 0, 1)
            
        change_heading = abs(avg_heading)    
        
        alpha = 2.0
        beta = 2.0

        #reward = change_heading#beta*manuever_reward #+ time_reward + change_heading
        
        other_agents = self.battle_space.agents
        for agent in other_agents:
            if agent.id != self.id and agent.is_pursuer:
                #compute the dot product
                dot_product = self.state_vector.dot_product_2D(agent.state_vector)
                # if self.is_colliding(self.radius_bubble):
                #     reward = -10.0
                #     break
        # reward = (alpha*distance_reward + 
        #           beta*manuever_reward + time_reward + change_heading)
        
        #we want to minimize the dot product so we add a negative
        # reward = -dot_product + distance_reward
        # reward = -(1 - closest_distance/self.max_distance_from)
        #clip the reward
        reward = distance_reward
        
        return reward
        
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
        action_cmd = self.action.copy()
        action_cmd[2] = -action_cmd[2]
        
        new_states = self.plane.rk45(
            self.plane.state_info, action_cmd, dt)
        self.on_state_update(new_states)

class DataHandler():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.u = []
        self.time = []
        self.rewards = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.roll.append(info_array[3])
        self.pitch.append(info_array[4])
        self.yaw.append(info_array[5])
        self.u.append(info_array[6])
        # self.time.append(info_array[7])
        
    def update_reward(self, reward:float) -> None:
        self.rewards.append(reward)
        
    def update_time(self, time:float) -> None:
        self.time.append(time)

class Plane():
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
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')
        self.v = ca.SX.sym('v')

        if self.include_time:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f, 
                self.v)
        else:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f,
                self.v 
            )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """
        controls for your system
        The controls are the roll, pitch, yaw, and airspeed
        If u_psi is 0 the plane will fly straight
        """
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """
        define the state space of your system
        NED Frame
        """
        self.g = 9.81 #m/s^2
        #body to inertia frame
        self.v_dot = (self.v_cmd - self.v)*(self.dt_val/self.airspeed_tau)
        self.x_fdot = self.v * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v * ca.sin(self.theta_f)
        
        self.phi_fdot   = (self.u_phi - self.phi_f) *(self.dt_val/self.pitch_tau)
        self.theta_fdot = (self.u_theta - self.theta_f) *(self.dt_val/self.pitch_tau) 
        
        #check if the denominator is zero
        #self.psi_fdot   = (self.u_psi * self.dt_val) + (self.g * (ca.tan(self.phi_f) / self.v_cmd))
        #self.psi_fdot =(self.u_psi * self.dt_val) + (self.g * ca.tan(self.phi_fdot) / self.v_cmd)
        self.psi_fdot = (self.g * ca.tan(self.phi_fdot) / self.v_cmd) + self.u_psi
        
        if self.include_time:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )
        else:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
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
        
        #clip the values of the angles
        next_step[3] = np.clip(next_step[3], 
                               -self.max_roll_rad, 
                               self.max_roll_rad)
        next_step[4] = np.clip(next_step[4], 
                               -self.max_pitch_rad, 
                               self.max_pitch_rad)
                       
        #wrap yaw from -pi to pi
        # if next_step[5] > np.pi:
        #     next_step[5] -= 2*np.pi
        # elif next_step[5] < -np.pi:
        #     next_step[5] += 2*np.pi
        # Wrap the angle
        
                    
        #clip the airspeed
        # next_step[6] = np.clip(next_step[6], 
        #                        self.min_airspeed_ms, 
        #                        self.max_airspeed_ms)            
                       
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            next_step[5] = (next_step[5] + np.pi) % (2 * np.pi) - np.pi
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    