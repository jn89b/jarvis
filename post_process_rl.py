# """
# Save this in a overall csv file so I can animate all the trajectories in one
# single go
# """
# import pandas as pd
# import numpy as np
# import pickle as pkl
# import glob as glob
# from jarvis.envs.simple_agent import DataHandler
# from matplotlib import pyplot as plt


# # get all pickle files in rl_pickle
# all_files = glob.glob("rl_pickle/*")
# print(all_files)

# evader_info = {
#     'x': [],
#     'y': [],
#     'z': []
# }

# pursuer_0_info = {
#     'x': [],
#     'y': [],
#     'z': []
# }

# pursuer_1_info = {
#     'x': [],
#     'y': [],
#     'z': []
# }


# # load all pickle files
# evader_win: int = 0
# for i, file in enumerate(all_files):
#     with open(file, 'rb') as f:
#         data = pkl.load(f)
#         print(data)
#         print("done")
#         data_info = data['datas']
#         for j, agent in enumerate(data_info):
#             agent: DataHandler
#             if j == 0:
#                 # save
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

#         if data['reward']['0'] > 1:
#             evader_win += 1


# total_files = len(all_files)
# win_rate = evader_win / total_files
# print("Win rate: ", win_rate)

# # convert everything into numpy arrays
# for key in evader_info.keys():
#     evader_info[key] = np.array(evader_info[key])

# for key in pursuer_0_info.keys():
#     pursuer_0_info[key] = np.array(pursuer_0_info[key])

# for key in pursuer_1_info.keys():
#     pursuer_1_info[key] = np.array(pursuer_1_info[key])

# # convert into pandas dataframe
# evader_df = pd.DataFrame(evader_info)
# pursuer_0_df = pd.DataFrame(pursuer_0_info)
# pursuer_1_df = pd.DataFrame(pursuer_1_info)

# # save into csv
# evader_df.to_csv("evader.csv")
# pursuer_0_df.to_csv("pursuer_0.csv")
# pursuer_1_df.to_csv("pursuer_1.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
matplotlib.rc('font', size=16)          # Base font
matplotlib.rc('axes', labelsize=18)     # X, Y labels
matplotlib.rc('axes', titlesize=22)     # Axes title
matplotlib.rc('xtick', labelsize=16)    # Tick labels
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('figure', titlesize=24)   # Figure "suptitle"
plt.close('all')

# Load CSV files (using index_col=0 to ignore the default index if present)
evader_df = pd.read_csv("rl_pickle/evader.csv", index_col=0)
pursuer_0_df = pd.read_csv("rl_pickle/pursuer_0.csv", index_col=0)
pursuer_1_df = pd.read_csv("rl_pickle/pursuer_1.csv", index_col=0)

# Extract the coordinates
evader_x = evader_df['x'].values
evader_y = evader_df['y'].values
evader_z = evader_df['z'].values

pursuer_0_x = pursuer_0_df['x'].values
pursuer_0_y = pursuer_0_df['y'].values
pursuer_0_z = pursuer_0_df['z'].values

pursuer_1_x = pursuer_1_df['x'].values
pursuer_1_y = pursuer_1_df['y'].values
pursuer_1_z = pursuer_1_df['z'].values

# Define colors: evader is blue; pursuers are variants of red.
evader_color = 'blue'
pursuer_colors = ['#ff9999', '#ff4d4d']  # light red and medium red

# Create the 3D figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Compute dynamic axis limits based on all data (with a 10% margin)
all_x = np.concatenate([evader_x, pursuer_0_x, pursuer_1_x])
all_y = np.concatenate([evader_y, pursuer_0_y, pursuer_1_y])
all_z = np.concatenate([evader_z, pursuer_0_z, pursuer_1_z])

x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
z_min, z_max = all_z.min(), all_z.max()

margin_x = 0.1 * (x_max - x_min)
margin_y = 0.1 * (y_max - y_min)
margin_z = 0.1 * (z_max - z_min)

ax.set_xlim(x_min - margin_x, x_max + margin_x)
ax.set_ylim(y_min - margin_y, y_max + margin_y)
ax.set_zlim(z_min - margin_z, z_max + margin_z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("50 Simulations of Evader vs Pursuers")

# Initialize line objects for the agents
evader_line, = ax.plot([], [], [], color=evader_color, label='Evader')
pursuer_0_line, = ax.plot(
    [], [], [], color=pursuer_colors[0], label='Pursuer 0')
pursuer_1_line, = ax.plot(
    [], [], [], color=pursuer_colors[1], label='Pursuer 1')
ax.legend()

# Use the minimum length of the coordinate arrays as the number of frames
n_frames = min(len(evader_x), len(pursuer_0_x), len(pursuer_1_x))


def init():
    # Clear all data from the lines
    evader_line.set_data([], [])
    evader_line.set_3d_properties([])
    pursuer_0_line.set_data([], [])
    pursuer_0_line.set_3d_properties([])
    pursuer_1_line.set_data([], [])
    pursuer_1_line.set_3d_properties([])
    return evader_line, pursuer_0_line, pursuer_1_line


def update(frame):
    # Optional trailing effect: show only the last 'trailing' steps
    trailing = 50
    start = max(0, frame - trailing)

    # Update evader trajectory
    evader_line.set_data(evader_x[start:frame], evader_y[start:frame])
    evader_line.set_3d_properties(evader_z[start:frame])

    # Update pursuer 0 trajectory
    pursuer_0_line.set_data(pursuer_0_x[start:frame], pursuer_0_y[start:frame])
    pursuer_0_line.set_3d_properties(pursuer_0_z[start:frame])

    # Update pursuer 1 trajectory
    pursuer_1_line.set_data(pursuer_1_x[start:frame], pursuer_1_y[start:frame])
    pursuer_1_line.set_3d_properties(pursuer_1_z[start:frame])

    return evader_line, pursuer_0_line, pursuer_1_line


# Create the animation
ani = FuncAnimation(fig, update, frames=n_frames,
                    init_func=init, interval=1, blit=True)

# tight layout
plt.tight_layout()
plt.show()
