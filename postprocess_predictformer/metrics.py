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

filename = "small_model"
fullname = filename+".pkl"
#info = pkl.load(open(os.path.join("postprocess_predictformer", "predictformer_output.pkl"), "rb"))
metrics = Metrics(fullname)
overall_metrics = metrics.predictformer_mse(
    slice_size=5)   
metrics.plot_mse_metrics(overall_metrics)
# save the metrics as a pickle file
pkl.dump(overall_metrics, open("ukf_metrics.pkl", "wb"))
 
info = pkl.load(open(fullname, "rb"))
center_gt_trajs:List[np.array] = info["center_gt_trajs"]
center_objects_world:List[np.array] = info["center_objects_world"]
predicted_probs: List[np.array] = [output['predicted_probability'] for output in info["output"]]
predicted_trajectories: List[np.array] = [output['predicted_ground_traj'] for output in info["output"]]
# predicted_trajectories: List[np.array] = [output['predicted_trajectory'] for output in info["output"]]
infer_time: List[float] = info["infer_time"]
total_steps = len(center_gt_trajs)
num_agents = center_gt_trajs[0].shape[1]
num_modes = predicted_probs[0].shape[1]
input_trajs = [output['input_obj_trajs'] for output in info["output"]]

# Define prediction horizon in seconds (adjust if dt != 1 second)
prediction_time = 6.0  # seconds
num_prediction_steps = int(prediction_time / 0.1)  # for dt=0.1, this equals 60

overall_ukf_estimates = []
overall_readings = []


for i in range(num_agents):
    ukf = UKFPlane()  # instantiate a new filter for each agent
    ukf_estimates = []
    ukf_readings = []
    for j in range(len(input_trajs)-1915):
        center_objects_world[j] = center_objects_world[j].squeeze()
        current_state = center_objects_world[j][i, 21, :]
        # Append an acceleration estimate (here assumed to be zero)
        current_state = np.concatenate([current_state, np.zeros(1)])
        
        # Initialize the UKF with the current state and update with measurement
        ukf.set_starting_state(current_state)
        z = get_z_measurement(current_state)
        ukf.update(z)
        
        # Predict for the next six seconds (i.e. num_prediction_steps)
        inner_ukf = UKFPlane()
        inner_ukf.set_starting_state(current_state)
        predictions = []
        
        for t in range(num_prediction_steps):
            z = get_z_measurement(ukf.get_estimate())
            # inner_ukf.run(z)
            inner_ukf.predict()
            inner_ukf.update(z)
            predictions.append(inner_ukf.ukf.x)  # use get_estimate() instead of ukf.state
            # print("prediction", inner_ukf.ukf.x)
            
        ukf_estimates.append(np.array(predictions))
        ukf_readings.append(current_state)
        print(f"Agent {i} Step {j} of {len(input_trajs)}")
        
    overall_ukf_estimates.append(ukf_estimates)
    overall_readings.append(ukf_readings)

#%%
# Let's plot each agent trajectory in a seperate plot and show the gaussian mixture model trajectory of the agent
# compute the mse for the ukf estimates against the ground truth for each agent
start_idx = 21
end_idx = start_idx + num_prediction_steps
overall_metrics = []
for i in range(num_agents):

    mse_metrics = {
        'overall_mse': [],
        'slice_mse': []
    }
    mse_total = 0.0
    slice_count = 0
    mse_total = 0.0
    ukf_traj = np.array(overall_ukf_estimates[i])
    
    for j in range(len(ukf_traj)):
        current_ukf = ukf_traj[j]
        ground_truth = center_objects_world[j][i,:,:]
        ground_truth = ground_truth[21:, :]
        # ground_truth_slice = ground_truth[start_idx:end_idx,0:3]
        # compute the slices of the ground truth and ukf estimates
        # num_steps = ground_truth_slice.shape[1]
        num_steps = ground_truth.shape[0]
        #num_steps = num_prediction_steps
        mse_bins = []
        slice_size:int = 1
        print("num_steps", num_steps)
        for step in range(0, num_steps, slice_size):
            print("step", step)
            pred_slice = current_ukf[step:step + slice_size, 0:3]
            gt_slice = ground_truth[step:step + slice_size, 0:3]
            slice_mse = np.mean((pred_slice - gt_slice) ** 2)
            mse_bins.append(slice_mse)
            mse_total += slice_mse
            slice_count += 1
            
        prev_mse = mse_bins[-1]
        mse_metrics['slice_mse'].append(mse_bins)
    
    
    ovearll_mse = mse_total / slice_count if slice_count > 0 else None
    # if j == 0:
    #     print(f"Agent {i} Overall MSE: {ovearll_mse}")
        
    mse_metrics['overall_mse'] = ovearll_mse
    overall_metrics.append(mse_metrics)
    
metrics.plot_mse_metrics(overall_metrics, to_break=False,
                         save_name="ukf_metrics", to_save=True)

fig, ax = plt.subplots(1, 1)
ax.plot(current_ukf[:, 0], current_ukf[:, 1], label='UKF')
ax.plot(ground_truth[:, 0], ground_truth[:, 1], label='True')
ax.legend()

# plot the 3D
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.plot(current_ukf[:, 0], current_ukf[:, 1], current_ukf[:, 2], label='UKF')
ax.plot(ground_truth[:end_idx, 0], ground_truth[:end_idx, 1], ground_truth[:end_idx, 2], label='True')
ax.legend()


# for i in range(len(overall_ukf_estimates)):
#     ukf_traj = np.array(overall_ukf_estimates[i])
#     ground_truth = center_objects_world[i][i,21:41,0:3]
#     mse = np.mean((ukf_traj - ground_truth)**2)
#     print(f"Agent {i} MSE: {mse}")
    

    