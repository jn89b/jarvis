import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D  
plt.close('all')
"""
Script to animate the trajectories of all the vehicles with the predictions

"""

#info = pkl.load(open(os.path.join("postprocess_predictformer", "predictformer_output.pkl"), "rb"))
info = pkl.load(open("predictformer_output.pkl", "rb"))

center_gt_trajs:List[np.array] = info["center_gt_trajs"]
center_objects_world:List[np.array] = info["center_objects_world"]
predicted_probs: List[np.array] = [output['predicted_probability'] for output in info["output"]]
predicted_trajectories: List[np.array] = [output['predicted_trajectory'] for output in info["output"]]
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
        center_gt_trajs[j] = center_gt_trajs[j].squeeze()
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


# Compute the mse of the predicted trajectory and the ground truth trajectory
def compute_mse(predicted_trajectory: np.array, ground_truth_trajectory: np.array, mask: np.array) -> float:
    """
    Compute the mean squared error between the predicted trajectory and the ground truth trajectory
    Args:
    predicted_trajectory: np.array: [num_modes, num_timesteps, num_attributes]
    ground_truth_trajectory: np.array: [num_timesteps, num_attributes]
    mask: np.array: [num_timesteps, 1]
    Returns:
    mse: float
    """
    num_timesteps = predicted_trajectory.shape[1]
    num_attributes = predicted_trajectory.shape[2]
    mse = 0.0
    for i in range(num_timesteps):
        for j in range(num_attributes):
            mse += np.sum((predicted_trajectory[:, i, j] - ground_truth_trajectory[i, j])**2)
    return mse


"""
Should I show the error propagation of the predicted trajectory?
Like the first 2 seconds how am I doing?
"""
delta_t: float = 0.1

slice_size = 10

overall_metrics = []

for j, agent in enumerate(overall_agents):
    predicted_trajectory = np.array(agent['predicted_trajectory'])
    ground_truth_trajectory = np.array(agent['ground_truth'])
    mse_total = 0.0
    slice_count = 0
    mse_metrics = {
        'overall_mse': [],
        'slice_mse': []
    }
    
    # Iterate over each trajectory for the agent
    for i in range(len(predicted_trajectory)):
        best_mode = np.argmax(agent['predicted_probability'][i])
        current_predicted_trajectory = predicted_trajectory[i]
        best_predicted_trajectory = current_predicted_trajectory[:, best_mode, :]
        
        # Slice ground truth from start_idx onward
        current_ground_truth_trajectory = ground_truth_trajectory[i][:, start_idx:]
        
        # Determine number of steps after start_idx
        num_steps = current_ground_truth_trajectory.shape[1]
        # Loop over time slices in steps of 10
        mse_bins = []
        for step in range(0, num_steps, slice_size):
            pred_slice = best_predicted_trajectory[:, step:step + slice_size]
            gt_slice = current_ground_truth_trajectory[:, step:step + slice_size]
            
            # Compute the MSE for the slice
            slice_mse = np.mean((pred_slice - gt_slice) ** 2)
            mse_bins.append(slice_mse)
            print(f"Agent {j}, trajectory {i}, slice {step}:{step+slice_size} MSE = {slice_mse}")
            
            mse_total += slice_mse
            slice_count += 1
        
        mse_metrics['slice_mse'].append(mse_bins)

    # Compute overall average MSE across all slices for the agent
    overall_mse = mse_total / slice_count if slice_count > 0 else None
    if j == 0:
        print(f"Agent {j} Overall Average MSE = {overall_mse}")
    mse_metrics['overall_mse'] = overall_mse
    overall_metrics.append(mse_metrics)
    
    
#%% 

# Example parameters (adjust as needed)
delta_t = 0.1
slice_size = 10
start_idx = 0

# # Use Seaborn Set1 palette
# sns.set_palette("Set1")

# Example parameters (adjust as needed)
delta_t = 0.1
slice_size = 10
start_idx = 0

# Determine the number of bins from the first agent's data
num_bins = len(np.mean(np.array(overall_metrics[0]['slice_mse']), axis=0))

# Create x positions linearly spaced from 1 to 6 seconds
x = np.linspace(1, 6, num=num_bins)

# Determine the width for grouped bars
num_agents = len(overall_metrics)
width = 0.8 / num_agents

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette("Set1")
colors = ["blue", "orange", "green"]
for i, agent in enumerate(overall_metrics):
    mse_bins = np.array(agent['slice_mse'])  # shape: (num_runs, num_bins)
    mean_mse = np.mean(mse_bins, axis=0)
    std_mse = np.std(mse_bins, axis=0)
    n = mse_bins.shape[0]
    sem = std_mse / np.sqrt(n)      # standard error of the mean
    ci = 1.96 * sem                 # 95% confidence interval

    # Offset each agent's bars so they appear side by side
    positions = x - 0.4 + width/2 + i * width
    ax.bar(positions, mean_mse, width=width, yerr=ci, capsize=5,
           label=f"Agent {i}", align='center', color=colors[i])

ax.set_xlabel("Projected Time (s)", fontsize=14)
ax.set_ylabel("Magnitude Distance MSE (m)", fontsize=14)
ax.set_title("MSE Error Propagation for Predicted Trajectories", fontsize=16)
ax.legend()
plt.tight_layout()

# save as an svg
plt.savefig("error_propagation_1.svg")

#compute the inference time mean and std
infer_time_mean = np.mean(infer_time)
infer_time_std = np.std(infer_time)
print(f"Mean inference time: {infer_time_mean} +/- {infer_time_std}")

plt.show()



