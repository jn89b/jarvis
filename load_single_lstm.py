import os
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from jarvis.transformers.wayformer.dataset import SingleLSTMDataset
from jarvis.transformers.traj_lstm import SingleAgentLSTMTrajectoryPredictor

plt.close('all')

# -------------------------
# Configuration & Device Setup
# -------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
config_path = "config/lstm_config.yaml"  # adjust path if needed
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# -------------------------
# Create Test Dataset & DataLoader
# -------------------------
# Here, we assume is_validation=True creates a test/validation dataset.
test_dataset = SingleLSTMDataset(config=config, is_test=True)
print("Test Dataset Length:", len(test_dataset))

dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

# -------------------------
# Locate Latest Checkpoint
# -------------------------
checkpoint_dir = "lstm_singletrajectory_checkpoint/"
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    ckpt_files = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime
    )
    if ckpt_files:
        latest_checkpoint = ckpt_files[-1]
        print("Resuming from checkpoint:", latest_checkpoint)
else:
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

# -------------------------
# Load the Trained LSTM Model
# -------------------------
model = SingleAgentLSTMTrajectoryPredictor.load_from_checkpoint(latest_checkpoint, config=config)
model.to(device)
model.eval()

# -------------------------
# Run Inference on a Few Batches
# -------------------------
output_history = []
infer_times = []

# Run inference on 6 batches for demonstration
for i, batch in enumerate(dataloader):
    # Move batch tensors to the proper device
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}

    start_time = time.time()
    
    output_traj, probs = model(batch)  # assume model returns (output, loss)
    end_time = time.time()
    print("inference time ", end_time - start_time)
    infer_times.append(end_time - start_time)
    validation_loss = model.validation_step(batch, i)
    output = {}
    output["predicted_trajectory"] = output_traj.detach().numpy().squeeze()
    #output['input_obj_trajs'] = batch['input_dict']['obj_trajs'].detach().numpy().squeeze()
    output['input_obj_trajs'] = batch['input_dict']['center_gt_trajs'].detach().numpy().squeeze()

    # print(f"Validation Loss for batch {i}: {validation_loss}")
    # output is originally size (batch_size, num_agents, num_modes, future_len, dims)
    #output = output.squeeze().detach().cpu().numpy()
    probs = probs.squeeze().detach().cpu().numpy()
    output_history.append(output)
    
    # if i == 10:
    #     break
    
    # # -------------------------
    # # Plot Predicted vs. Ground Truth Trajectories
    # # -------------------------
    # # For demonstration, we plot the trajectories for the first batch.
    # first_output = output_history[0]
    
    # # Assume the batch dictionary has an 'input_dict' key with ground truth trajectories.
    # # These keys may need adjustment based on your dataset/model.
    # # Here we assume:
    # #  - 'predicted_trajectory' is shaped [num_agents, num_timesteps, dims]
    # #  - 'center_gt_trajs' under batch['input_dict'] is the ground truth trajectory
    # batch_ground_truth = batch['input_dict']['center_gt_trajs'].squeeze().detach().cpu().numpy()
    # predicted = first_output['predicted_trajectory'].detach().cpu().numpy()  # shape: [num_agents, num_timesteps, dims]
    # predicted = predicted[0,0,:,:]
    # probs = first_output['probs']  # shape: [num_agents, num_modes]
    # num_agents = predicted.shape[0]
    
    # #%%
    # for j in range(1):
    #     if i <= 10:
    #         break
    #     # plot the predicted trajectory
    #     fig, ax = plt.subplots()
    #     gt_traj = batch_ground_truth[:, 0:3]
    #     ax.plot(gt_traj[:, 0], gt_traj[:, 1], label="Ground Truth", linewidth=2)
    #     ax.plot(predicted[ :, 0], predicted[ :, 1], 'o--', label="Predicted")
    #     # num_modes = first_output['probs'].shape[0]
    #     # for mode in range(num_modes):
    #     #     pred_traj = predicted[agent, mode, :, 0:3]
    #     #     prob = probs[agent, mode]
    #     #     ax.plot(pred_traj[:, 0], pred_traj[:, 1], label=f"Predicted Mode {mode} (Prob: {prob:.2f})")
    #         # ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'o--', label=f"Predicted Mode {mode}")
    #     ax.set_title(f"Agent {i} Trajectory")
    #     ax.legend()
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
        
    # plt.show()

# for agent in range(num_agents):
#     fig, ax = plt.subplots()
#     # Ground truth trajectory: assuming first two dimensions are x and y coordinates
#     gt_traj = batch_ground_truth[agent, :, 0:2]
#     ax.plot(gt_traj[:, 0], gt_traj[:, 1], label="Ground Truth", linewidth=2)
    
#     # Predicted trajectory: plotted with dashed line and circle markers
#     pred_traj = predicted[agent, :, 0:2]
#     ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'o--', label="Predicted")
    
#     ax.set_title(f"Agent {agent} Trajectory")
#     ax.legend()
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")

# plt.show()

import pickle as pkl
info = {"output": output_history,
        "infer_time": infer_times}
folder_dir = "postprocess_predictformer"
if not os.path.exists(folder_dir):
    os.makedirs(folder_dir)
pkl.dump(info, open(os.path.join(folder_dir, "lstm_small_model.pkl"), "wb"))
