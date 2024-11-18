from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from jarvis.transformers.evadeformer import EvadeFormer
from jarvis.datasets.base_dataset import BaseDataset
import yaml
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
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
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
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
    ground_truth_uncentered = vals['uncentered_trajs_future']
    # Run the model to get predictions
    with torch.no_grad():
        init_time = time.time()
        output, _ = model(batch)  # Assuming model returns output and loss
        final_time = time.time() - init_time
        print("final time", final_time)
        # Shape: (batch_size, T, 2)
        predicted_trajectories = output['predicted_trajectory']
        predicted_probability = output['predicted_probability']
        decoder_attn = output['decoder_attention']
        encoder_attn = output['encoder_attention']
        # we are compressing from the 4 attention by computing the mean
        # Averaging across heads
        avg_attn = [layer.mean(dim=1) for layer in decoder_attn]
        avg_attn = [layer.cpu().numpy() for layer in avg_attn]
        # Summing across embedding dimension
        temporal_attn = [layer.sum(dim=-1) for layer in decoder_attn]
        temporal_attn = [layer.cpu().numpy() for layer in temporal_attn]
        avg_attention = encoder_attn[0].mean(
            dim=-1).cpu().detach().numpy()  # Shape: (3, 192)

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
    ground_truth_uncentered = ground_truth_uncentered.cpu().numpy()

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot each trajectory in the batch
    # Loop through each sample in the batch
    # Define colors for the pursuers
    pursuer_colors = ["green", "orange", "purple", "teal"]

    # Loop through each sample in the batch
    # Define colors for the pursuers
    pursuer_colors = ["green", "orange", "purple", "teal"]

    # # Loop through each agent in the batch
    # for agent_idx in range(decoder_attn[0].shape[0]):  # Assuming decoder_attn[0] has shape (3, 4, 64, 192)
    #     for layer_idx, layer in enumerate(decoder_attn):  # Loop through each layer in decoder_attn
    #         for head_idx in range(layer.shape[1]):  # Loop through each head in the layer

    #             # Extract attention weights for the specific agent, layer, and head
    #             attn_weights = layer[agent_idx, head_idx].cpu().detach().numpy()  # Shape (64, 192)

    #             # Create a heatmap
    #             plt.figure(figsize=(10, 6))
    #             sns.heatmap(attn_weights, cmap="viridis")
    #             plt.title(f"Attention Weights for Agent {agent_idx}, Layer {layer_idx}, Head {head_idx}")
    #             plt.xlabel("Embedding Dimension")
    #             plt.ylabel("Timestep")

    # for agent_idx in range(avg_attention.shape[0]):
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(avg_attention[agent_idx], label=f"Agent {agent_idx}")
    #     plt.xlabel("Position (Token Index)")
    #     plt.ylabel("Average Attention Value")
    #     plt.title(f"Encoder Attention Across Positions for Agent {agent_idx}")
    #     plt.legend()
    #     plt.show()

    # Loop through each sample in the batch
    for i in range(gt_trajectories.shape[0]):
        # Plot evader (index 0)
        x_init = ground_truth_uncentered[i, 0, 0]
        y_init = ground_truth_uncentered[i, 0, 1]
        if i == 0:
            print(x_init, y_init)
            plt.plot(gt_trajectories[i, :, 0]+x_init, gt_trajectories[i, :, 1]+y_init,
                     color="blue", alpha=0.6, linestyle="-", label="Evader GT")
            plt.plot(pred_trajectories[i, :, 0]+x_init, pred_trajectories[i, :, 1]+y_init,
                     color="red", alpha=0.6, linestyle="--", label="Evader Predicted")
            # Start position marker for evader
            plt.scatter(gt_trajectories[i, 0, 0]+x_init, gt_trajectories[i, 0, 1]+y_init,
                        color="blue", edgecolor="black", s=50, label="Evader Start")
        else:
            # Plot pursuers with different colors
            color = pursuer_colors[(i - 1) % len(pursuer_colors)]
            plt.plot(gt_trajectories[i, :, 0]+x_init, gt_trajectories[i, :, 1]+y_init,
                     color=color, alpha=0.6, linestyle="-", label=f"Pursuer {i} GT")
            plt.plot(pred_trajectories[i, :, 0]+x_init, pred_trajectories[i, :, 1]+y_init,
                     color=color, alpha=0.6, linestyle="--", label=f"Pursuer {i} Predicted")
            # Start position marker for each pursuer
            plt.scatter(gt_trajectories[i, 0, 0]+x_init, gt_trajectories[i, 0, 1]+y_init,
                        color=color, edgecolor="black", s=50, label=f"Pursuer {i} Start")

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
