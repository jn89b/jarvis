import matplotlib.pyplot as plt
import numpy as np
from typing import Type as Tuple
from jarvis.envs.battle_space import BattleSpace
from jarvis.assets.Plane import Agent
from jarvis.assets.Radar2D import Radar2D, RadarSystem2D
class Visualizer(object):
    def __init__(self) -> None:
        # this is for the 3D animation
        self.lines = []
        self.scatters = []
        self.min_x = -1000 
        self.max_x = 1000
        self.min_y = -1000
        self.max_y = 1000
        
    def plot_2d_trajectory(self, battlespace:BattleSpace,
                           use_own_fig:bool=False,
                           fig=None, ax=None) -> Tuple:
        
        if not use_own_fig:
            fig, ax = plt.subplots()
        
        for agent in battlespace.agents:
            agent: Agent
            data = agent.plane.data_handler
            if agent.is_pursuer:
                color = 'r'
            else:
                color = 'b'
                
            x = data.x
            y = data.y
            ax.scatter(x[0], y[0], color='g', label='Start')
            ax.plot(x, y, color=color, 
                    label=agent.id, 
                    linestyle='-', marker='o', markersize=2)
            #print the length of the trajectory
                        
            #set titles
            ax.set_title('2D Trajectory for agent {}'.format(agent.id))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        return fig, ax        
    
    def plot_3d_trajectory(self, battlespace:BattleSpace) -> Tuple:
        # Plot 3D trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for agent in battlespace.agents:
            agent: Agent
            data = agent.plane.data_handler
            if agent.is_pursuer:
                color = 'r'
            else:
                color = 'b'
                print("Data: ", data.x[-1], data.y[-1])
                
            x = data.x
            y = data.y
            z = data.z
            ax.scatter(x[0], y[0], z[0], color='g', label='Start')
            ax.plot(x, y, z, color, label=agent.id)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # SHOW LEGEND
        ax.legend()
        return fig, ax
    
    def plot_attitudes(self, battlespace:BattleSpace) -> Tuple:
        # Plot attitudes
        for agent in battlespace.agents:
            fig, ax = plt.subplots(4, 1)
            if agent.is_pursuer:
                color = 'r'
            else:
                color = 'b'
                
            ax[0].plot(np.rad2deg(agent.plane.data_handler.roll), color, label='phi')
            ax[1].plot(np.rad2deg(agent.plane.data_handler.pitch), color, label='theta')    
            ax[2].plot(np.rad2deg(agent.plane.data_handler.yaw), color, label='psi')
            ax[3].plot(agent.plane.data_handler.u, color, label='airspeed')
            
            #set titles
            ax[0].set_title('Attitudes for agent {}'.format(agent.id))
            ax[0].set_ylabel('phi')
            ax[1].set_ylabel('theta')
            ax[2].set_ylabel('psi')
            ax[3].set_ylabel('airspeed')
            
        return fig, ax
    
    def plot_attitudes2d(self, battlespace:BattleSpace, ignore_pursuer:bool=False) -> Tuple:
        for agent in battlespace.agents:
            
            if ignore_pursuer and agent.is_pursuer:
                continue
            
            fig, ax = plt.subplots(2, 1)
            if agent.is_pursuer:
                color = 'r'
            else:
                color = 'b'
                
            #plot the yaw and velocity
            ax[0].plot(np.rad2deg(agent.plane.data_handler.yaw), color, label='psi')
            ax[1].plot(agent.plane.data_handler.u, color, label='airspeed')
            
            ax[0].set_title('Yaw and Airspeed for agent {}'.format(agent.id))
            ax[0].set_ylabel('Yaw')
            ax[1].set_ylabel('Airspeed')
            
        return fig, ax
    
    def plot_radars(self, battlespace:BattleSpace) -> Tuple:
        # Plot the radar
        fig, ax = plt.subplots()
        
        radar_system = battlespace.radar_system
        for radar in radar_system.radars:
            radar: Radar2D
            x = radar.position[0]
            y = radar.position[1]
            id = radar.radar_id
            ax.scatter(x, y, color='r', label='Radar {}'.format(id))
            #plot circle
            circle = plt.Circle((x, y), radar.range, color='r', fill=True,
                                alpha=0.5)
            
            ax.add_artist(circle)
            
        #plot the trajectory as well
        self.plot_2d_trajectory(battlespace, fig=fig, ax=ax, use_own_fig=True)
        #set limits 
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_ylim([self.min_y, self.max_y])
                        
        return fig, ax