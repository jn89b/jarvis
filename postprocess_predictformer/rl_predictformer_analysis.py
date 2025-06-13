import pickle as pkl 
import os
import numpy as np
import matplotlib.pyplot as plt
import einops

from jarvis.utils.sim_helper import transform_with_current_heading

def simple_transform(pred_traj: np.ndarray,
                     current_heading: float,
                     current_position: np.ndarray):
    """
    pred_traj:      shape (num_modes, T, >=2), where [.,.,0:2] are local‐frame x,y
    current_heading: scalar, in radians
    current_position: array([cx, cy]), shape (2,)
    Returns (global_traj_x, global_traj_y) each of shape (num_modes, T).
    """
    dx = pred_traj[:, :, 0]   # (num_modes, T)
    dy = pred_traj[:, :, 1]   # (num_modes, T)

    ch = np.cos(current_heading)
    sh = np.sin(current_heading)

    # rotate:
    global_dx = dx * ch - dy * sh
    global_dy = dx * sh + dy * ch

    # translate:
    cx, cy = current_position
    global_x = global_dx + cx
    global_y = global_dy + cy

    return global_x, global_y

plt.close('all')
# import pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pkl.load(file)
    return data


# Loading observation history 
data = load_pickle_file('pursuer_evader_history.pkl')
observation_history = data['observation_history']
ego_history = []
pursuer_history = []

for d in observation_history:
    ego_history.append(d['ego'])
    pursuer_history.append(d['vehicles'])
    
# ground_truth_history = data['batch_history']
# ground_truth = []
# for gt in ground_truth_history:
#     gt = gt.squeeze()
#     ground_truth.append(gt)
    
original_pos_history = data['original_pos_history']

ego_history = np.array(ego_history)
pursuer_history = np.array(pursuer_history)
# move pursuer history around
pursuer_history = einops.rearrange(pursuer_history, 'n v a -> v n a')
predicted_trajectory = data['predicted_traj']

num_agents: int = 2

#%%
"""
The questions I have right now is, how well am I actually predicting the trajectory of the pursuers?
- Are they within some distrubution of the ground truth of the trajectories it should be?
- To find out I need to compare the 
"""
# let's plot each agent trajectory in a separate plot and show the predicted trajectory of the agent 

idx = 200
desired_traj = predicted_trajectory[idx]
current_idx:int = 20
original_pos = original_pos_history[idx][:,current_idx,:]
buffer_idx: int = 25
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(ego_history[:idx+buffer_idx, 0], ego_history[:idx+buffer_idx, 1], label='Ego Vehicle', color='blue')
ax.scatter(ego_history[0, 0], ego_history[0, 1], label='Ego Start', color='blue', marker='o')

for i in range(num_agents):
    # # Plot the ego vehicle trajectory
    # #ego_traj = ego_history[:, i, :2]  # Assuming ego history has shape (timesteps, agents, attributes)
    # ego_traj = ego_history[:,:2]
    # ax.plot(ego_traj[:, 0], ego_traj[:, 1], label='Ego Vehicle', color='blue', marker='o')
    
    # Plot the pursuer trajectories
    pursuer_traj = pursuer_history[i]

    ax.plot(pursuer_traj[:idx+buffer_idx, 0], pursuer_traj[:idx+buffer_idx, 1], label=f'Pursuer {i}')
    ax.scatter(pursuer_traj[0, 0], pursuer_traj[0, 1], label=f'Pursuer {i} Start', marker='o')
    # Plot the predicted trajectory
    # pred_traj = predicted_trajectory[i]
    #ax.plot(pred_traj[:, 0], pred_traj[:, 1], label='Predicted Trajectory', color='red', linestyle='--')
    

for i in range(desired_traj.shape[0]):
    if i == 0:
        continue
    pred_traj = desired_traj[i]
    current_pursuer_traj = pursuer_history[i-1]
    x_start = current_pursuer_traj[idx, 0]
    y_start = current_pursuer_traj[idx, 1]
    current_heading = original_pos[i-1, 5]
    #current_heading = current_pursuer_traj[idx, 5]  # Assuming heading is at index 5
    # desired_ground_truth = ground_truth[idx][i]
    current_position = current_pursuer_traj[idx, :2]
    print(f"Current Position for Pursuer {i}: {current_position}")
    print("current heading", current_heading)
    # print(f"Current Heading for Pursuer {i}: {np.rad2deg(current_heading)} degrees")
    print("actual heading", np.rad2deg(current_pursuer_traj[idx, 5]))
    transformed_traj = transform_with_current_heading(
        pred_traj=pred_traj,
        current_heading=current_heading,
        current_position=current_position,
        heading_index=5
    )
    for j in range(transformed_traj.shape[0]):
        x = transformed_traj[j, :, 0] #+ x_start
        y = transformed_traj[j, :, 1] #+ y_start

        ax.plot(x, y, label=f'Predicted Trajectory {i}', linestyle='--')
    
    # let's plot what is being fed into the transformer from the ground truth
    
    # x_gt = desired_ground_truth[:, 0]
    # y_gt = desired_ground_truth[:, 1]
    # ax.plot(x_gt, y_gt, label=f'Ground Truth {i}', color='cyan', marker='x')
    
    
ax.set_title(f'Agent {i+1} Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend()

plt.show()
