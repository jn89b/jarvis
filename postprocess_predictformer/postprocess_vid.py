import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D  
# plt.close('all')
"""
- Script to animate the trajectories of all the vehicles with the predictions

"""

#info = pkl.load(open(os.path.join("postprocess_predictformer", "predictformer_output.pkl"), "rb"))
info = pkl.load(open("noisy_predictformer_output_2.pkl", "rb"))

center_gt_trajs:List[np.array] = info["center_gt_trajs"]
center_objects_world:List[np.array] = info["center_objects_world"]
predicted_probs: List[np.array] = [output['predicted_probability'] for output in info["output"]]
#predicted_trajectories: List[np.array] = [output['predicted_trajectory'] for output in info["output"]]
predicted_trajectories: List[np.array] = [output['predicted_ground_traj'] for output in info["output"]]
infer_time: List[float] = info["infer_time"]

total_steps = len(center_gt_trajs)
num_agents = center_gt_trajs[0].shape[1]
num_modes = predicted_probs[0].shape[1]

# Let's plot each agent trajectory in a seperate plot and show the gaussian mixture model trajectory of the agent
# create a data structure to store the data of each agents:
# ground truth position 
# trajectory predictions for each gaussian
# probability of each gaussian
# and then animate
overall_agents = []
start_idx = 21
for i in range(num_agents):
    agent = {}
    overall_positions = []
    predicted_trajs = []
    predicted_modes = []
    x_history = []
    y_history = []
    z_history = []
    for j in range(total_steps):
        # the ground truth trajectory is a [num_agents, num_timesteps, num_attributes]
        center_objects_world[j] = center_objects_world[j].squeeze()
        predicted_trajectories[j] = predicted_trajectories[j].squeeze()
        predicted_probs[j] = predicted_probs[j].squeeze()
        x_gt = center_objects_world[j][i,:, 0]
        y_gt = center_objects_world[j][i,:, 1]
        z_gt = center_objects_world[j][i,:, 2]
        # the predicted trajectory is [num_agents, num_modes, num_timesteps, num_attributes]
        x_pred = predicted_trajectories[j][i,:,:, 0] + x_gt[start_idx]
        y_pred = predicted_trajectories[j][i,:,:, 1] + y_gt[start_idx]
        z_pred = predicted_trajectories[j][i,:,:, 2] + z_gt[start_idx]
        x_history.extend(x_gt)
        y_history.extend(y_gt)
        z_history.extend(z_gt)
        overall_positions.append([x_gt, y_gt, z_gt])
        predicted_trajs.append([x_pred, y_pred, z_pred])
        predicted_modes.append(predicted_probs[j][i])

    x_positions = np.array([pos[0] for pos in overall_positions])
    y_positions = np.array([pos[1] for pos in overall_positions])
    z_positions = np.array([pos[2] for pos in overall_positions])
    x_positions = x_positions.flatten()
    y_positions = y_positions.flatten()
    z_positions = z_positions.flatten()
    agent['position'] = [x_positions, y_positions, z_positions]
    agent['ground_truth'] = overall_positions
    agent['predicted_trajectory'] = predicted_trajs
    agent['predicted_probability'] = predicted_modes
    overall_agents.append(agent)

def animate_agent_3d(agent, num_modes, id_num):
    total_steps = len(agent['ground_truth'])
    print(total_steps)
    
    # Set up the 3D figure and axis.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # get the max and min of the x, y, z positions
    x_max = np.max(agent['position'][0])
    x_min = np.min(agent['position'][0])
    y_max = np.max(agent['position'][1])
    y_min = np.min(agent['position'][1])
    z_max = np.max(agent['position'][2])
    z_min = np.min(agent['position'][2])
    
    buffer = 10
    
    # Set fixed plot limits; adjust these limits as needed based on your data.
    ax.set_xlim(xmin=x_min, xmax= x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_title("Agent ID: " + str(id_num))
    
    # Create a line for the ground truth trajectory.
    gt_line, = ax.plot([], [], [], 'b-', label='Ground Truth')
    
    # Use a Seaborn color palette for predicted modes.
    palette = sns.color_palette("deep", num_modes)  # Returns a list of RGB tuples.
    pred_lines = []
    for mode in range(num_modes):
        line, = ax.plot([], [], [], linestyle='--', color=palette[mode],
                        label=f'Predicted Mode {mode+1}')
        pred_lines.append(line)

    # Create a marker for the initial (or current) position.
    init_point, = ax.plot([], [], [], 'ko', markersize=8, label='Initial Position')
    
    # Avoid duplicate legends if multiple modes exist.
    ax.legend(loc='upper right')
    
    def init():
        gt_line.set_data([], [])
        gt_line.set_3d_properties([])
        for line in pred_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        init_point.set_data([], [])
        init_point.set_3d_properties([])
        return [gt_line, init_point] + pred_lines

    def update(frame):
        # Update ground truth for current frame.
        gt = agent['ground_truth'][frame]  # [x_gt, y_gt, z_gt]
        x_gt, y_gt, z_gt = gt
        buffer = 1
        gt_line.set_data(x_gt[start_idx:], y_gt[start_idx:])
        gt_line.set_3d_properties(z_gt[start_idx:])
        
        # Update predicted trajectories.
        # We assume predicted_trajectory is stored as [x_pred, y_pred, z_pred].
        pred_data = agent['predicted_trajectory'][frame]
        # Check if multiple modes exist by inspecting the dimensionality.
        if np.ndim(pred_data[0]) > 1:
            # For each mode, update its corresponding line.
            for mode in range(num_modes):
                x_pred = pred_data[0][mode]
                y_pred = pred_data[1][mode]
                z_pred = pred_data[2][mode]
                pred_lines[mode].set_data(x_pred, y_pred)
                pred_lines[mode].set_3d_properties(z_pred)
        else:
            # For a single mode, update only the first line.
            pred_lines[0].set_data(pred_data[0], pred_data[1])
            pred_lines[0].set_3d_properties(pred_data[2])
            # Clear any additional lines (if they were created).
            for mode in range(1, num_modes):
                pred_lines[mode].set_data([], [])
                pred_lines[mode].set_3d_properties([])
        
        # Mark the initial position (e.g., the first point in the ground truth for that frame).
        init_point.set_data(x_gt[0:1], y_gt[0:1])
        init_point.set_3d_properties(z_gt[0:1])
        return [gt_line, init_point] + pred_lines

    ani = animation.FuncAnimation(fig, update, frames=total_steps,
                                  init_func=init, blit=True, interval=5)
    return ani


def animate_agents_together(agents, num_modes):
    # Assume all agents have the same number of time steps.
    total_steps = len(agents[0]['ground_truth'])
    
    # Compute global axis limits from all agents' ground truth.
    all_x, all_y, all_z = [], [], []
    for agent in agents:
        for frame in agent['ground_truth']:
            x_gt, y_gt, z_gt = frame  # Each frame is [x_gt, y_gt, z_gt] (each an array)
                        
            all_x.extend(x_gt)
            all_y.extend(y_gt)
            all_z.extend(z_gt)
    # Add a buffer for visualization.
    buffer = 10
    x_min, x_max = np.min(all_x) - buffer, np.max(all_x) + buffer
    y_min, y_max = np.min(all_y) - buffer, np.max(all_y) + buffer
    z_min, z_max = np.min(all_z) - buffer, np.max(all_z) + buffer

    # Set up the figure and 3D axis.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_title("All Agents 3D Trajectories")
    
    # Create a palette for agents.
    agent_palette = sns.color_palette("colorblind", len(agents))
    
    # Containers for line objects.
    gt_lines = []         # Ground truth lines for each agent.
    init_points = []      # Initial position markers for each agent.
    pred_lines_all = []   # Predicted mode lines for each agent (list of lists).
    
    for i, agent in enumerate(agents):
        agent_color = agent_palette[i]
        
        # Ground truth line (solid line).
        gt_line, = ax.plot([], [], [], '-', color=agent_color,
                           label=f'Agent {i+1} Ground Truth')
        gt_lines.append(gt_line)
        
        # Initial position marker.
        init_point, = ax.plot([], [], [], 'o', color=agent_color,
                              markersize=8, label=f'Agent {i+1} Initial')
        init_points.append(init_point)
        
        # Predicted modes lines (dashed lines).
        # Use a light palette based on the agent color.
        mode_palette = sns.light_palette(agent_color, n_colors=num_modes)
        agent_pred_lines = []
        for mode in range(num_modes):
            pred_line, = ax.plot([], [], [], '--', color=mode_palette[mode],
                                 label=f'Agent {i+1} Pred Mode {mode+1}')
            agent_pred_lines.append(pred_line)
        pred_lines_all.append(agent_pred_lines)
    
    # Create a legend (note that with many agents/modes the legend might be crowded).
    ax.legend(loc='upper right', fontsize='small')

    def init():
        artists = []
        for line in gt_lines:
            line.set_data([], [])
            line.set_3d_properties([])
            artists.append(line)
        for point in init_points:
            point.set_data([], [])
            point.set_3d_properties([])
            artists.append(point)
        for agent_pred_lines in pred_lines_all:
            for line in agent_pred_lines:
                line.set_data([], [])
                line.set_3d_properties([])
                artists.append(line)
        return artists

    def update(frame):
        artists = []
        # Update each agent.
        for i, agent in enumerate(agents):
            # Ground truth update.
            gt = agent['ground_truth'][frame]  # [x_gt, y_gt, z_gt]
            x_gt, y_gt, z_gt = gt
            buffer = -1
            gt_line.set_data(x_gt[start_idx:], y_gt[start_idx:])
            gt_line.set_3d_properties(z_gt[start_idx:])
            artists.append(gt_lines[i])
            
            # Predicted trajectories update.
            pred_data = agent['predicted_trajectory'][frame]  # [x_pred, y_pred, z_pred]
            # Check if multiple modes exist.
            if np.ndim(pred_data[0]) > 1:
                for mode in range(num_modes):
                    x_pred = pred_data[0][mode]
                    y_pred = pred_data[1][mode]
                    z_pred = pred_data[2][mode]
                    pred_lines_all[i][mode].set_data(x_pred, y_pred)
                    pred_lines_all[i][mode].set_3d_properties(z_pred)
                    artists.append(pred_lines_all[i][mode])
            else:
                # Single mode case.
                pred_lines_all[i][0].set_data(pred_data[0], pred_data[1])
                pred_lines_all[i][0].set_3d_properties(pred_data[2])
                artists.append(pred_lines_all[i][0])
                for mode in range(1, num_modes):
                    pred_lines_all[i][mode].set_data([], [])
                    pred_lines_all[i][mode].set_3d_properties([])
                    artists.append(pred_lines_all[i][mode])
            
            # Update initial position marker (using the first point of ground truth for the current frame).
            init_points[i].set_data(x_gt[start_idx:start_idx+1], y_gt[start_idx:start_idx+1])
            init_points[i].set_3d_properties(z_gt[start_idx:start_idx+1])
            artists.append(init_points[i])
        return artists

    # For 3D animations, blitting is typically disabled.
    ani = animation.FuncAnimation(fig, update, frames=total_steps,
                                  init_func=init, blit=True, interval=10)
    return ani

# Example usage:
# Assume overall_agents is a list containing 3 agent dictionaries,
# each with keys 'ground_truth' and 'predicted_trajectory' (and optionally 'position').
# Also assume that each agent's 'ground_truth' is a list of frames,
# where each frame is [x_gt, y_gt, z_gt] and each is an array.
# And each agent's 'predicted_trajectory' follows a similar structure.

# For example:
# overall_agents = [agent1, agent2, agent3]
# num_modes = overall_agents[0]['predicted_probability'][0].shape[0] if np.ndim(overall_agents[0]['predicted_probability'][0]) > 0 else 1

# Create the animation and keep a reference so it’s not garbage-collected.
together_ani = animate_agents_together(overall_agents, num_modes)


# Example usage:
# Determine the number of modes from the predicted probability structure.
# If the probability is stored as an array for each agent and frame,
# we assume a single mode if it’s not an array with more than one dimension.
# num_modes = overall_agents[0]['predicted_probability'][0].shape[0] if np.ndim(overall_agents[0]['predicted_probability'][0]) > 0 else 1
animations = []
# Create an animation for each agent.
for i,agent in enumerate(overall_agents):
    ani = animate_agent_3d(agent, num_modes, i)
    animations.append(ani)

animations.append(together_ani)

plt.show()
