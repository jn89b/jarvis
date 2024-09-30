import matplotlib.pyplot as plt
import numpy as np
from typing import Type as Tuple
from jarvis.envs.battle_space import BattleSpace
from jarvis.assets.Plane import Agent
from jarvis.assets.Radar2D import Radar2D, RadarSystem2D
from matplotlib.animation import FuncAnimation


class Visualizer(object):
    def __init__(self) -> None:
        # this is for the 3D animation
        self.lines = []
        self.scatters = []
        self.min_x = -1000 
        self.max_x = 1000
        self.min_y = -1000
        self.max_y = 1000
        
    def plot_2d_trajectory(self, battlespace: BattleSpace,
                        use_own_fig: bool = False,
                        fig=None, ax=None) -> Tuple:
        
        if not use_own_fig:
            fig, ax = plt.subplots()
        
        lines = []
        scatter_start = []
        
        for agent in battlespace.agents:
            agent: Agent
            data = agent.plane.data_handler
            color = 'r' if agent.is_pursuer else 'b'
            
            # Initialize the line and scatter point
            line, = ax.plot([], [], color=color, label=agent.id, linestyle='-', marker='o', markersize=2)
            # Set the initial scatter point at the agent's starting position
            scatter = ax.scatter([data.x[0]], [data.y[0]], color='g', label='Start')
            
            lines.append(line)
            scatter_start.append(scatter)

        ax.set_title('2D Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        ax.set_xlim(-2000, 2000)  # Adjust limits based on your space
        ax.set_ylim(-2000, 2000)
        
        def init():
            # Initialize each line and scatter point
            for line in lines:
                line.set_data([], [])
            for scatter in scatter_start:
                scatter.set_offsets([[], []])
            return lines + scatter_start

        def update(frame):
            for i, agent in enumerate(battlespace.agents):
                data = agent.plane.data_handler
                
                # Update the line with agent's trajectory up to current frame
                x = data.x[:frame]
                y = data.y[:frame]
                lines[i].set_data(x, y)
                
                # Keep scatter point at the initial position
                scatter_start[i].set_offsets([x[0], y[0]])
            
            return lines + scatter_start

        # Set blit=False to avoid issues with certain backends
        ani = FuncAnimation(fig, update, frames=np.arange(1, len(battlespace.agents[0].plane.data_handler.x)), 
                            init_func=init, blit=False, repeat=False)
    
    def animate_2d_trajectory(self, battlespace: BattleSpace,
                        use_own_fig: bool = False,
                        fig=None, ax=None) -> Tuple:
        
        if not use_own_fig:
            fig, ax = plt.subplots()
        
        # Initialize empty lists to store the plot objects for each agent
        lines = []
        scatter_start = []
        
        # Initialize the plot for each agent
        for agent in battlespace.agents:
            agent: Agent
            data = agent.plane.data_handler
            if agent.is_pursuer:
                color = 'r'
            else:
                color = 'b'
            
            # Set up an empty line and scatter for start position
            line, = ax.plot([], [], color=color, label=agent.id, 
                            linestyle='-', marker='o', markersize=2)
            scatter = ax.scatter([], [], color='g', label='Start')
            
            lines.append(line)
            scatter_start.append(scatter)

        # Set titles and labels
        ax.set_title('2D Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Set axis limits (you can adjust these based on your environment)
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        
        # Function to initialize the plot
        def init():
            for line in lines:
                line.set_data([], [])
            for scatter in scatter_start:
                scatter.set_offsets([[], []])
            return lines + scatter_start


        print("lines: ", lines)
        # Function to update the plot in each frame
        def update(frame):
            for i, agent in enumerate(battlespace.agents):
                data = agent.plane.data_handler
                
                # Update trajectory
                x = data.x[:frame]  # Use up to current frame data
                y = data.y[:frame]
                lines[i].set_data(x, y)
                
                # Update start point (if needed, stays the same)
                scatter_start[i].set_offsets([x[0], y[0]])
            
            return lines + scatter_start

        # Create animation
        ani = FuncAnimation(fig, update, frames=np.arange(1, 
                            len(battlespace.agents[0].plane.data_handler.x)), 
                            init_func=init, blit=True, repeat=False)

        return fig, ax, ani
    
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