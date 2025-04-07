import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import pandas as pd
from typing import List, Tuple
# from mpl_toolkits.mplot3d import Axes3D  
from jarvis.utils.metrics import Metrics
from jarvis.algos.filters import UKFPlane
plt.close('all')

def get_z_measurement(state:np.array) -> np.array:
    """
    Get a noisy measurement from the true state
    """
    X_IDX:int = 0
    Y_IDX:int = 1
    Z_IDX:int = 2
    PSI_IDX:int = 5
    VEL_IDX:int = 6

    # pos_noise = 0.1
    # psi_noise = 0.1
    # vel_noise = 0.1
    num_measurements = 5
    measurements:np.array = np.zeros(num_measurements)
    measurements[0] = state[X_IDX] 
    measurements[1] = state[Y_IDX] 
    measurements[2] = state[Z_IDX] 
    measurements[3] = state[PSI_IDX] 
    measurements[4] = state[VEL_IDX]
    
    return measurements

# """
# Script to animate the trajectories of all the vehicles with the prediction
# """

filename = "lstm_small_model"
fullname = filename+".pkl"
#info = pkl.load(open(os.path.join("postprocess_predictformer", "predictformer_output.pkl"), "rb"))
# metrics = Metrics(fullname)
# overall_metrics = metrics.predictformer_mse(
#     slice_size=5)   
# metrics.plot_mse_metrics(overall_metrics)
# # save the metrics as a pickle file
# pkl.dump(overall_metrics, open("ukf_metrics.pkl", "wb"))
 
info = pkl.load(open(fullname, "rb"))
# center_gt_trajs:List[np.array] = info["center_gt_trajs"]
# center_objects_world:List[np.array] = info["center_objects_world"]
# predicted_probs: List[np.array] = [output['predicted_probability'] for output in info["output"]]
predicted_trajectories: List[np.array] = [output['predicted_trajectory'] for output in info["output"]]
# predicted_trajectories: List[np.array] = [output['predicted_trajectory'] for output in info["output"]]
infer_time: List[float] = info["infer_time"]
# total_steps = len(center_gt_trajs)
# num_agents = center_gt_trajs[0].shape[1]
#num_modes = predicted_probs[0].shape[1]
input_trajs = [output['input_obj_trajs'] for output in info["output"]]
# Define prediction horizon in seconds (adjust if dt != 1 second)
prediction_time = 6.0  # seconds
num_prediction_steps = int(prediction_time / 0.1)  # for dt=0.1, this equals 60

#%%
slice_size:int = 5
start_idx:int = 0
for i in range(len(input_trajs)):
    # input_trajs[i] = input_trajs[i].squeeze()
    mse_total = 0.0
    slice_count = 0
    mse_metrics = {
        'overall_mse': [],
        'slice_mse': []
    }
    ground_truth_trajectory = input_trajs[i]
    predicted_trajectory = predicted_trajectories[i]
    prev_mse: float = 0.0
    for j in range(len(predicted_trajectories)):        
        #best_modes = np.argsort(agent['predicted_probability'][j])[-3:]
        current_predicted_trajectory = predicted_trajectories[j]
        lowest_mse = np.inf
        best_mode:int = 0
                
        # Slice ground truth from start_idx onward
        current_ground_truth_trajectory = ground_truth_trajectory[:, start_idx:]
        
        # Determine number of steps after start_idx
        num_steps = current_ground_truth_trajectory.shape[0]
        # Loop over time slices in steps of 10s
        mse_bins = []
        for step in range(0, num_steps, slice_size):
            pred_slice = current_predicted_trajectory[step:step + slice_size, 0:3]
            gt_slice = current_ground_truth_trajectory[step:step + slice_size, 0:3]
            
            # Compute the MSE for the slice
            #slice_mse = np.mean((pred_slice - gt_slice) ** 2)
            # compute mean absolute error
            slice_mse = np.abs(pred_slice - gt_slice)
            slice_mse = np.sum(slice_mse)/gt_slice.shape[1]
            # slice_mse = (slice_mse)
            mse_bins.append(slice_mse)
            
            mse_total += slice_mse + prev_mse
            slice_count += 1
        
        prev_mse = mse_bins[-1]
        mse_metrics['slice_mse'].append(mse_bins)

    # Compute overall average MSE across all slices for the agent
    overall_mse = mse_total / slice_count if slice_count > 0 else None
    if j == 0:
        print(f"Agent {j} Overall Average MSE = {overall_mse}")
    mse_metrics['overall_mse'] = overall_mse
    #overall_metrics.append(mse_metrics)

# save the mse values
import pickle as pkl
folder_dir = "postprocess_predictformer"
if not os.path.exists(folder_dir):
    os.makedirs(folder_dir)
pkl.dump(mse_metrics, open(os.path.join(folder_dir, "lstm_metrics.pkl"), "wb"))

# for i in range(num_agents):
#     ukf = UKFPlane()  # instantiate a new filter for each agent
#     ukf_estimates = []
#     ukf_readings = []
#     for j in range(len(input_trajs)-1915):
#         center_objects_world[j] = center_objects_world[j].squeeze()
#         current_state = center_objects_world[j][i, 21, :]
#         # Append an acceleration estimate (here assumed to be zero)
#         current_state = np.concatenate([current_state, np.zeros(1)])
        
#         # Initialize the UKF with the current state and update with measurement
#         ukf.set_starting_state(current_state)
#         z = get_z_measurement(current_state)
#         ukf.update(z)
        
#         # Predict for the next six seconds (i.e. num_prediction_steps)
#         inner_ukf = UKFPlane()
#         inner_ukf.set_starting_state(current_state)
#         predictions = []
        
#         for t in range(num_prediction_steps):
#             z = get_z_measurement(ukf.get_estimate())
#             # inner_ukf.run(z)
#             inner_ukf.predict()
#             inner_ukf.update(z)
#             predictions.append(inner_ukf.ukf.x)  # use get_estimate() instead of ukf.state
#             # print("prediction", inner_ukf.ukf.x)
            
#         ukf_estimates.append(np.array(predictions))
#         ukf_readings.append(current_state)
#         print(f"Agent {i} Step {j} of {len(input_trajs)}")
        
#     overall_ukf_estimates.append(ukf_estimates)
#     overall_readings.append(ukf_readings)


