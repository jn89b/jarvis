from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from jarvis.transformers.evadeformer import EvadeFormer
from jarvis.datasets.base_dataset import BaseDataset
import yaml
import matplotlib.pyplot as plt
import os
# Path to the best model checkpoint
# checkpoint_path = "checkpoints/evadeformer-epoch=176-val_loss=0.52.ckpt"

# Check if there's an existing checkpoint to resume from
checkpoint_dir = "checkpoints/"
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f)
         for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")

model_config = 'config/data_config.yaml'
with open(model_config, 'r') as f:
    model_config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model from the checkpoint
model = EvadeFormer.load_from_checkpoint(
    checkpoint_path=latest_checkpoint, hparams_file='config/data_config.yaml',
    config=model_config)
model.to("cpu")  # Move the model to the same device as the data
model.eval()  # Set the model to evaluation mode


# Load the data configuration
data_config = "config/data_config.yaml"
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)
batch_size: int = 5
dataset = BaseDataset(config=data_config, is_validation=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=dataset.collate_fn)


all_mse_errors = []

# Loop over test batches
for batch in dataloader:
    # Extract ground truth trajectories from the batch
    # Shape: (batch_size, T, 2)
    vals = batch['input_dict']
    ground_truth_trajectories = vals['center_gt_trajs']
    # Move data to the same device as the model
    batch = {k: v.to(model.device) if isinstance(
        v, torch.Tensor) else v for k, v in batch.items()}

    ground_truth_test = torch.cat([vals['center_gt_trajs'][..., :2], vals['center_gt_trajs_mask'].unsqueeze(-1)],
                                  dim=-1)

    # Run the model to get predictions
    with torch.no_grad():
        output, _ = model(batch)  # Assuming model returns output and loss
        # Shape: (batch_size, T, 2)
        predicted_trajectories = output['predicted_trajectory']
        predicted_probability = output['predicted_probability']

    # # Convert predictions and ground truth to numpy for comparison
    # predicted_trajectories = predicted_trajectories.cpu().numpy()
    # ground_truth_trajectories = ground_truth_trajectories.cpu().numpy()

    # # Calculate MSE for each trajectory in the batch
    # print(predicted_trajectories)
    # Assume predicted_probability has shape [batch_size, num_modes]
    # and predicted_trajectories has shape [batch_size, num_modes, timesteps, params]

    # Get the index of the most probable mode for each sample in the batch
    most_probable_mode = predicted_probability.argmin(dim=1)  # Shape: [15]

    # Select the trajectory corresponding to the most probable mode
    # predicted_trajectories has shape [15, 6, 60, 5]
    best_predicted_trajectories = predicted_trajectories[
        torch.arange(predicted_trajectories.size(0)), most_probable_mode
    ]  # Shape: [15, 60, 5]

    # Extract only x and y coordinates from the predicted trajectories
    predicted_xy = best_predicted_trajectories[..., :2]  # Shape: [15, 60, 2]

# Convert tensors to numpy for easier plotting, if they are not already
gt_trajectories = ground_truth_trajectories.cpu().numpy()
pred_trajectories = predicted_xy.cpu().numpy()
predicted_xy = predicted_xy.cpu().numpy()

# Create the plot
plt.figure(figsize=(10, 8))

# Plot each trajectory in the batch
# Loop through each sample in the batch
for i in range(gt_trajectories.shape[0]):
    plt.plot(gt_trajectories[i, :, 0], gt_trajectories[i, :, 1],
             color="blue", alpha=0.6, linestyle="--", label="GT" if i == 0 else "")
    plt.plot(pred_trajectories[i, :, 0], pred_trajectories[i, :, 1],
             color="red", alpha=0.6, linestyle="--", label="Predicted" if i == 0 else "")

# Add labels and title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Overall Trajectory Comparison: Ground Truth vs Predicted")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Ensures equal scaling for accurate spatial comparison

plt.show()

# # Calculate the average MSE across all trajectories
# average_mse = sum(all_mse_errors) / len(all_mse_errors)
# print(f"Average MSE for the test dataset: {average_mse}")
