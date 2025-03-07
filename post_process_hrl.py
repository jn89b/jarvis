from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import glob
import os
import seaborn as sns
from jarvis.envs.simple_agent import DataHandler
from typing import List
matplotlib.rc('font', size=16)
matplotlib.rc('axes', labelsize=18)
matplotlib.rc('axes', titlesize=22)
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('figure', titlesize=24)
plt.close('all')


# %% Multiple process data for mean win and std -----------------------------------
all_folders = glob.glob("hrl_data/*/", recursive=True)
win_history = []
# loop through all folders and get pickle files
for folder in all_folders:
    # get all pickle files in rl_picklecentral plot
    evader_win: int = 0
    pickle_files = glob.glob(folder + "/*")
    for i, file in enumerate(pickle_files):
        with open(file, 'rb') as f:
            data = pkl.load(f)
            print(data)
            print("done")
            data_info = data['datas']
            for j, agent in enumerate(data_info):
                agent: DataHandler
            if data['reward']['good_guy_hrl'] > 1:
                evader_win += 1

    win_rate = evader_win / len(pickle_files)
    win_history.append(win_rate)

# For each folder, the bad guy win rate is simply (1 - win_rate)
bad_win_history = [1 - wr for wr in win_history]

# %%
# Create a DataFrame for plotting
df_good = pd.DataFrame({"Win Rate": win_history, "Group": "Good Guy (HRL)"})
df_bad = pd.DataFrame({"Win Rate": bad_win_history, "Group": "Pursuers"})
df = pd.concat([df_good, df_bad], ignore_index=True)

# Define custom palette: good guy in blue, bad guy in red
custom_palette = {"Good Guy (HRL)": "blue", "Pursuers": "red"}

# Plot a violin plot with custom colors
plt.figure(figsize=(6.5, 8))
sns.violinplot(x="Group", y="Win Rate", data=df, palette=custom_palette)
plt.title("HRL Win Rates with Different Seeds")
# save svg file
plt.tight_layout()
plt.savefig("figures/hrl_win_rate.svg")
plt.show()


# %%
# # Create directory for saving CSV files if it doesn't exist
# if not os.path.exists("rl_pickle"):
#     os.makedirs("rl_pickle")

# # Get all pickle files from the 'hrl_data' folder
# all_files = glob.glob("hrl_data/*")
# print("Found files:", all_files)

# # Initialize dictionaries to store data for each agent and goal
# evader_info = {'x': [], 'y': [], 'z': []}
# pursuer_0_info = {'x': [], 'y': [], 'z': []}
# pursuer_1_info = {'x': [], 'y': [], 'z': []}
# goal_info = {'x': [], 'y': [], 'z': []}

# evader_win: int = 0

# # Process each pickle file
# for i, file in enumerate(all_files):
#     with open(file, 'rb') as f:
#         data = pkl.load(f)
#         data_info = data['datas']
#         # Extract the goal location from the first element of goal_history
#         goal_x_val = data['goal_history'][0].x
#         goal_y_val = data['goal_history'][0].y
#         goal_z_val = data['goal_history'][0].z
#         print("Goal location:", goal_x_val, goal_y_val, goal_z_val)

#         # Loop through the agent data and store trajectories
#         for j, agent in enumerate(data_info):
#             if j == 0:
#                 evader_info['x'].extend(agent.x)
#                 evader_info['y'].extend(agent.y)
#                 evader_info['z'].extend(agent.z)
#             elif j == 1:
#                 pursuer_0_info['x'].extend(agent.x)
#                 pursuer_0_info['y'].extend(agent.y)
#                 pursuer_0_info['z'].extend(agent.z)
#             elif j == 2:
#                 pursuer_1_info['x'].extend(agent.x)
#                 pursuer_1_info['y'].extend(agent.y)
#                 pursuer_1_info['z'].extend(agent.z)

#         # Count wins (example condition)
#         if data['reward']['good_guy_hrl'] > 1:
#             evader_win += 1

#         # For the goal, replicate the same location for each time point in the simulation.
#         num_points = len(data_info[0].x)
#         goal_info['x'].extend([goal_x_val] * num_points)
#         goal_info['y'].extend([goal_y_val] * num_points)
#         goal_info['z'].extend([goal_z_val] * num_points)

# total_files: int = len(all_files)
# win_rate: float = evader_win / total_files if total_files > 0 else 0
# print("Win rate:", win_rate)

# # Convert lists to numpy arrays for each agent and the goal
# for key in evader_info:
#     evader_info[key] = np.array(evader_info[key])
# for key in pursuer_0_info:
#     pursuer_0_info[key] = np.array(pursuer_0_info[key])
# for key in pursuer_1_info:
#     pursuer_1_info[key] = np.array(pursuer_1_info[key])
# for key in goal_info:
#     goal_info[key] = np.array(goal_info[key])

# # Convert the numpy arrays to pandas DataFrames
# evader_df = pd.DataFrame(evader_info)
# pursuer_0_df = pd.DataFrame(pursuer_0_info)
# pursuer_1_df = pd.DataFrame(pursuer_1_info)
# goal_df = pd.DataFrame(goal_info)

# # Save DataFrames as CSV files in the "rl_pickle" folder
# evader_df.to_csv("rl_pickle/evader.csv")
# pursuer_0_df.to_csv("rl_pickle/pursuer_0.csv")
# pursuer_1_df.to_csv("rl_pickle/pursuer_1.csv")
# goal_df.to_csv("rl_pickle/goal.csv")

# # Configure matplotlib fonts and sizes

# # Load the CSV files (using index_col=0 to ignore the default index)
# evader_df = pd.read_csv("rl_pickle/evader.csv", index_col=0)
# pursuer_0_df = pd.read_csv("rl_pickle/pursuer_0.csv", index_col=0)
# pursuer_1_df = pd.read_csv("rl_pickle/pursuer_1.csv", index_col=0)
# goal_df = pd.read_csv("rl_pickle/goal.csv", index_col=0)

# # Extract coordinates for each agent and the goal
# evader_x = evader_df['x'].values
# evader_y = evader_df['y'].values
# evader_z = evader_df['z'].values

# pursuer_0_x = pursuer_0_df['x'].values
# pursuer_0_y = pursuer_0_df['y'].values
# pursuer_0_z = pursuer_0_df['z'].values

# pursuer_1_x = pursuer_1_df['x'].values
# pursuer_1_y = pursuer_1_df['y'].values
# pursuer_1_z = pursuer_1_df['z'].values

# goal_x = goal_df['x'].values
# goal_y = goal_df['y'].values
# goal_z = goal_df['z'].values

# # Define colors for the agents
# evader_color = 'blue'
# pursuer_colors = ['#ff9999', '#ff4d4d']  # light red and medium red

# # Create a 3D figure and axis
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Compute axis limits based on all data and add a margin
# all_x = np.concatenate([evader_x, pursuer_0_x, pursuer_1_x])
# all_y = np.concatenate([evader_y, pursuer_0_y, pursuer_1_y])
# all_z = np.concatenate([evader_z, pursuer_0_z, pursuer_1_z])
# x_min, x_max = all_x.min(), all_x.max()
# y_min, y_max = all_y.min(), all_y.max()
# z_min, z_max = all_z.min(), all_z.max()
# margin_x = 0.1 * (x_max - x_min)
# margin_y = 0.1 * (y_max - y_min)
# margin_z = 0.1 * (z_max - z_min)

# # Set custom axis limits (adjust if necessary)
# # min_x, max_x = -600, 600
# # min_y, max_y = -600, 600
# ax.set_xlim(x_min - margin_x, x_max + margin_x)
# ax.set_ylim(y_min - margin_y, y_max + margin_y)
# # ax.set_zlim(z_min - margin_z, z_max + margin_z)
# ax.set_zlim(20, 80)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Hiearchial RL vs Pursuers")

# # Initialize line objects for the agents
# evader_line, = ax.plot([], [], [], color=evader_color,
#                        label='Evader', linewidth=3)
# pursuer_0_line, = ax.plot(
#     [], [], [], color=pursuer_colors[0], label='Pursuer 0', linewidth=3)
# pursuer_1_line, = ax.plot(
#     [], [], [], color=pursuer_colors[1], label='Pursuer 1', linewidth=3)

# # Create a scatter plot for the goal location.
# # If the goal is dynamic, we'll update its position each frame.
# goal_scatter = ax.scatter(goal_x[0], goal_y[0], goal_z[0],
#                           s=200, color='green', marker='*',
#                           label='Goal')
# ax.legend()

# # Determine the number of frames based on the shortest trajectory
# n_frames = min(len(evader_x), len(pursuer_0_x), len(pursuer_1_x), len(goal_x))


# def init():
#     # Initialize agent trajectories
#     evader_line.set_data([], [])
#     evader_line.set_3d_properties([])
#     pursuer_0_line.set_data([], [])
#     pursuer_0_line.set_3d_properties([])
#     pursuer_1_line.set_data([], [])
#     pursuer_1_line.set_3d_properties([])
#     return evader_line, pursuer_0_line, pursuer_1_line, goal_scatter


# def update(frame):
#     trailing = 50  # Adjust trailing length for a trailing effect
#     start = max(0, frame - trailing)

#     # Update trajectories for each agent
#     evader_line.set_data(evader_x[start:frame], evader_y[start:frame])
#     evader_line.set_3d_properties(evader_z[start:frame])

#     pursuer_0_line.set_data(pursuer_0_x[start:frame], pursuer_0_y[start:frame])
#     pursuer_0_line.set_3d_properties(pursuer_0_z[start:frame])

#     pursuer_1_line.set_data(pursuer_1_x[start:frame], pursuer_1_y[start:frame])
#     pursuer_1_line.set_3d_properties(pursuer_1_z[start:frame])

#     # Update the goal scatter position.
#     # Option 1: using lists
#     goal_scatter._offsets3d = (
#         [goal_x[frame]], [goal_y[frame]], [goal_z[frame]])

#     # Option 2 (if Option 1 doesn't work, try setting blit=False in FuncAnimation)
#     return evader_line, pursuer_0_line, pursuer_1_line, goal_scatter


# plt.tight_layout()

# # Note: If the goal scatter still doesn't update correctly, try setting blit=False.
# ani = FuncAnimation(fig, update, frames=n_frames,
#                     init_func=init, interval=5, blit=True)

# plt.show()
