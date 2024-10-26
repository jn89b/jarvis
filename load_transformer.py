import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Tuple
# from jarvis.transformers.test import CarTransformer
from jarvis.transformers.evader_transformer import EvaderTransformer
from jarvis.transformers.data_loader import (
    CarTrajectoryDataset, collate_fn, DataLoader, open_json_file)

"""
Random notes
the attention weights tructure is a tuple of 8 since there are 8 layers of attention
- If index to one of the values it is size:
        - [batch_size, num_heads, seq_length, seq_length]
        - num_heads = number of attention heads so 8
        - seq_length is the total number of tokens so if we have 1 ego and 2 vehicles will be 3

"""


def get_ego_attention_weights(
        attn_weights: torch.Tensor, layer: int = 0) -> torch.TensorType:
    """
    Get the attention weights for the ego vehicle
    """
    ego_attn_weights = attn_weights[layer][0, :, 0, :]
    return ego_attn_weights


def extract_avg_attention_with_ego(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Extracts the average attention from the ego vehicle (token 0) to itself and other vehicles.

    Args:
        attention_weights (torch.Tensor): Attention weights from the transformer.
                                          Shape: (batch_size, num_heads,
                                                  seq_length, seq_length)

    Returns:
        torch.Tensor: Average attention from ego to itself and other vehicles. Shape: (batch_size, seq_length)
    """
    avg_attention_per_layer = []
    for layer_attention in attention_weights:
        ego_attention_all_tokens = layer_attention[:, :, 0, :]
        avg_attention_all_tokens = ego_attention_all_tokens.mean(dim=1)
        avg_attention_per_layer.append(avg_attention_all_tokens)

    avg_attention_stacked = torch.stack(avg_attention_per_layer, dim=0)
    avg_attention = avg_attention_stacked.mean(dim=0)

    # convert to numpy cpu
    return avg_attention.cpu().numpy()


def extract_avg_attention_to_vehicles(attention_weights: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    Extract and average attention from the ego vehicle to all other vehicles across all layers.

    Args:
        attention_weights (Tuple[torch.Tensor]): Attention weights from all layers,
                                                 each with shape (batch_size, num_heads, seq_length, seq_length).

    Returns:
        torch.Tensor: Average attention from ego to other vehicles across all layers
                      and heads, with shape (batch_size, num_vehicles).
    """
    # List to store averaged attention from each layer
    avg_attention_per_layer = []

    # Iterate through each layer in the attention weights
    for layer_attention in attention_weights:
        # Extract attention from ego (token 0) to all vehicles (tokens 1 onward)
        # Shape: (batch_size, num_heads, num_vehicles)
        ego_attention_to_vehicles = layer_attention[:, :, 0, 1:]

        # Average the attention across heads for this layer
        # Shape: (batch_size, num_vehicles)
        avg_attention_to_vehicles_layer = ego_attention_to_vehicles.mean(dim=1)

        # Store the averaged attention for this layer
        avg_attention_per_layer.append(avg_attention_to_vehicles_layer)

    # Stack the averaged attention from each layer
    # Shape: (num_layers, batch_size, num_vehicles)
    avg_attention_stacked = torch.stack(avg_attention_per_layer, dim=0)

    # Finally, average across layers
    # Shape: (batch_size, num_vehicles)
    avg_attention_to_vehicles = avg_attention_stacked.mean(dim=0)

    # convert to numpy cpu
    return avg_attention_to_vehicles.cpu().numpy()


def get_avg_attention_per_vehicle(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Averages attention weights across heads and layers to get attention values for each vehicle.

    Args:
        attn_weights (torch.Tensor): Attention weights from all layers,
                                     shape [num_layers, batch_size, num_heads, num_vehicles].

    Returns:
        torch.Tensor: Average attention per vehicle, shape [batch_size, num_vehicles].
    """
    # Average across heads (dimension 2) to get [num_layers, batch_size, num_vehicles]
    avg_head_attention = attn_weights.mean(dim=2)

    # Average across layers (dimension 0) to get [batch_size, num_vehicles]
    avg_layer_attention = avg_head_attention.mean(dim=0)

    return avg_layer_attention.cpu().numpy()


def get_pursuer_distances(vehicle_data: torch.Tensor) -> torch.Tensor:
    """
    Computes the norm distance
    """
    pursuers = vehicle_data.squeeze()
    pursuers = pursuers[:, :2]
    # Compute L2 norm across the rows
    distances = torch.norm(pursuers, p=2, dim=1)

    return distances.cpu().numpy()


def get_pursuer_heading(vehicle_data: torch.Tensor) -> torch.Tensor:
    """
    Computes the heading of the pursuers
    """
    pursuers = vehicle_data.squeeze()
    pursuers = pursuers[:, 2]
    # Compute L2 norm across the rows

    return pursuers.cpu().numpy()

# Example usage:
# attention_weights = output.attentions  # Extracted from the transformer model's output
# avg_attention = extract_avg_attention_to_vehicles(attention_weights)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load multiple JSON files for the dataset
file_lists = []

for data in range(1, 4):
    file_name = 'data/' + 'simulation_data_' + str(data) + '.json'
    json_data = open_json_file(file_name)
    file_lists.append(json_data)

# print(f"Loaded {len(file_lists)} data files")
dataset = CarTrajectoryDataset(file_lists)
test_dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# Initiis alize the model architecture
model = EvaderTransformer().to(device)

# Load the saved model parameters (weights and biases)
# model.load_state_dict(torch.load('evader_transformer_model_100.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
checkpoint_path = 'evader_transformer_model.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
print(f"Resuming training from epoch {start_epoch}")


# Make sure to set the model in evaluation mode
model.eval()

# Define loss function and optimizer
test_loss_fn = torch.nn.L1Loss()
# Calculate total loss on the test set
total_test_loss = 0.0

# Run inference loop without calculating gradients
ground_truth_list = []
predicted_list = []
pursuer_1_attention = []
pursuer_2_attention = []
pursuer_1_distance = []
pursuer_2_distance = []
pursuer_1_heading = []
pursuer_2_heading = []

with torch.no_grad():
    for batch in test_dataloader:
        # Extract input data
        padded_ego, padded_vehicles, waypoints, idx = batch
        padded_ego = padded_ego.to(device)
        padded_vehicles = padded_vehicles.to(device)
        waypoints = waypoints.to(device)

        seq_length = padded_ego.shape[1]

        for t in range(0, 100):
            ego_data_t = padded_ego[:, t, :]
            vehicles_data_t = padded_vehicles[:, t, :, :]
            waypoints_t = waypoints[:, t, :]
            last_wp = waypoints_t[0, -1, :]

            # Get predictions
            pred_waypoints, attn_weights = model(
                vehicles_data_t, waypoints_t, last_wp)

            # print(f"Predicted waypoints: {pred_waypoints}")
            # print(f"Actual waypoints: {waypoints_t}")
            # print("\n")
            # Compute the loss
            loss = test_loss_fn(pred_waypoints, waypoints_t)
            total_test_loss += loss.item()
            ground_truth_list.append(waypoints_t.cpu().numpy().squeeze(0))
            predicted_list.append(pred_waypoints.cpu().numpy().squeeze(0)[1:])
            attention_weight = extract_avg_attention_with_ego(attn_weights)
            # attention_weight = get_avg_attention_per_vehicle(attn_weights)
            distances = get_pursuer_distances(vehicles_data_t)
            headings = get_pursuer_heading(vehicles_data_t)
            pursuer_1_distance.append(distances[0])
            pursuer_2_distance.append(distances[1])
            pursuer_1_attention.append(attention_weight[0][0])
            pursuer_2_attention.append(attention_weight[0][1])
            pursuer_1_heading.append(np.rad2deg(headings[0]))
            pursuer_2_heading.append(np.rad2deg(headings[1]))
            # print("ego_data_t: ", ego_data_t)
            # print(
            #     f"Predicted waypoints: {pred_waypoints.cpu().numpy().squeeze(0)[1:]}")
            # print(f"Actual waypoints: {waypoints_t}")
            # print("\n")
# Calculate average test loss
avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Average test loss: {avg_test_loss:.4f}")


def plot_trajectory(ground_truth, predicted, title="Trajectory Prediction"):
    plt.figure(figsize=(8, 6))
    plt.plot(ground_truth[:, 0], ground_truth[:, 1],
             label='Ground Truth', color='green')
    plt.plot(predicted[:, 0], predicted[:, 1],
             label='Predicted', color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


gx_list = []
gy_list = []
px_list = []
py_list = []
for g, p in zip(ground_truth_list, predicted_list):
    gx_list.extend(g[:, 0])
    gy_list.extend(g[:, 1])
    last_row = p[0]
    px_list.append(last_row[0])
    py_list.append(last_row[1])
    # px_list.extend(p[:, 0])
    # py_list.extend(p[:, 1])


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(gx_list, gy_list, label='Ground Truth', color='green')
ax.scatter(gx_list[0], gy_list[0], label='Start', color='blue', s=100)
ax.plot(px_list, py_list, label='Predicted', color='red', linestyle='dashed')
ax.scatter(px_list[0], py_list[0], label='Start', color='blue', s=100)
ax.set_title('Trajectory Prediction')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pursuer_1_attention, label='Pursuer 1 Attention', color='blue')
ax.plot(pursuer_2_attention, label='Pursuer 2 Attention', color='red')
ax.set_title('Average Attention to Pursuers')
ax.set_xlabel('Time Step')
ax.set_ylabel('Attention')
ax.legend()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pursuer_1_distance, label='Pursuer 1 Distance', color='blue')
ax.plot(pursuer_2_distance, label='Pursuer 2 Distance', color='red')
ax.set_title('Distance to Evader')
ax.set_xlabel('Time Step')
ax.set_ylabel('Distance')
ax.legend()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(pursuer_1_heading, label='Pursuer 1 Heading', color='blue')
ax.plot(pursuer_2_heading, label='Pursuer 2 Heading', color='red')
ax.set_title('Heading of Pursuers')
ax.set_xlabel('Time Step')
ax.set_ylabel('Heading')
ax.legend()

plt.show()

# # Example: Plot predicted vs ground truth waypoints
# plot_trajectory(ground_truth_list, predicted_list)
