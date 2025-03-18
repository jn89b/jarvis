import os
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from jarvis.transformers.wayformer.dataset import LSTMDataset
from jarvis.transformers.traj_lstm import MultiAgentLSTMTrajectoryPredictor

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
test_dataset = LSTMDataset(config=config, is_validation=True)
print("Test Dataset Length:", len(test_dataset))

dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=test_dataset.collate_fn
)

# -------------------------
# Locate Latest Checkpoint
# -------------------------
checkpoint_dir = "lstm_multi_trajectory_checkpoint/"
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
model = MultiAgentLSTMTrajectoryPredictor.load_from_checkpoint(latest_checkpoint, config=config)
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
    output, probs = model(batch)  # assume model returns (output, loss)
    end_time = time.time()
    infer_times.append(end_time - start_time)
    # output is originally size (batch_size, num_agents, num_modes, future_len, dims)
    output = output.squeeze().detach().cpu().numpy()
    probs = probs.squeeze().detach().cpu().numpy()
    output_np = {
        "predicted_trajectory": output,
        "probs": probs
    }
    # for key, value in output.items():
    #     if isinstance(value, torch.Tensor):
    #         output_np[key] = value.detach().cpu().numpy()
    #     else:
    #         output_np[key] = value
    output_history.append(output_np)

    print(f"Inference time for batch {i}: {end_time - start_time:.4f} seconds")
    if i == 5:
        break

# -------------------------
# Plot Predicted vs. Ground Truth Trajectories
# -------------------------
# For demonstration, we plot the trajectories for the first batch.
first_output = output_history[0]

# Assume the batch dictionary has an 'input_dict' key with ground truth trajectories.
# These keys may need adjustment based on your dataset/model.
# Here we assume:
#  - 'predicted_trajectory' is shaped [num_agents, num_timesteps, dims]
#  - 'center_gt_trajs' under batch['input_dict'] is the ground truth trajectory
batch_ground_truth = batch['input_dict']['center_gt_trajs'].squeeze().detach().cpu().numpy()
predicted = first_output['predicted_trajectory']  # shape: [num_agents, num_timesteps, dims]
probs = first_output['probs']  # shape: [num_agents, num_modes]
num_agents = predicted.shape[0]

for agent in range(num_agents):
    # plot the predicted trajectory
    fig, ax = plt.subplots()
    gt_traj = batch_ground_truth[agent, :, 0:3]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], label="Ground Truth", linewidth=2)
    ax.plot(predicted[agent, :, 0], predicted[agent, :, 1], 'o--', label="Predicted")
    # num_modes = first_output['probs'].shape[0]
    # for mode in range(num_modes):
    #     pred_traj = predicted[agent, mode, :, 0:3]
    #     prob = probs[agent, mode]
    #     ax.plot(pred_traj[:, 0], pred_traj[:, 1], label=f"Predicted Mode {mode} (Prob: {prob:.2f})")
        # ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'o--', label=f"Predicted Mode {mode}")
    ax.set_title(f"Agent {agent} Trajectory")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
plt.show()

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
