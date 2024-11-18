from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import os
import matplotlib
import time
import json

from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from jarvis.transformers.evaderformer2 import EvaderFormer
from jarvis.datasets.base_dataset import PlanTDataset, UAVTDataset
plt.close('all')


class AgentHistory():
    def __init__(self) -> None:
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self.psi: List[float] = []
        self.v: List[float] = []
        self.waypoints_x: List[float] = []
        self.waypoints_y: List[float] = []
        self.pred_waypoints_x: List[float] = []
        self.pred_waypoints_y: List[float] = []
        self.attention_scores: List[float] = []


class JSONData():
    def __init__(self, json_filename: str) -> None:
        self.json_filename: str = json_filename
        self.data: Dict = self.load_json()
        self.ego: AgentHistory = AgentHistory()
        self.pursuer_1: AgentHistory = AgentHistory()
        self.pursuer_2: AgentHistory = AgentHistory()
        for d in self.data:
            ego_data = d['ego']
            pursuer_data = d['vehicles']
            self.ego.x.append(ego_data[0])
            self.ego.y.append(ego_data[1])
            self.ego.psi.append(ego_data[2])
            self.ego.v.append(ego_data[3])

            for i, v in enumerate(pursuer_data):
                if i == 0:
                    self.pursuer_1.x.append(v[0])
                    self.pursuer_1.y.append(v[1])
                else:
                    self.pursuer_2.x.append(v[0])
                    self.pursuer_2.y.append(v[1])

    def load_json(self) -> Dict:
        with open(self.json_filename, 'r') as f:
            data = json.load(f)
        return data


def compute_attention_scores(attention_map: Tuple[torch.tensor]) -> np.ndarray:
    """
    Compute the normalized attention scores from the cls token
    """
    # Shape: [batch_size, sequence_length]
    relevance_scores = torch.zeros(batch_size, num_tokens)

    # Sum attention weights across all layers and heads
    for layer_attention in attention_map:
        # Sum over heads to get the attention distribution of the [CLS] token across the sequence
        cls_attention = layer_attention[:, :, 0, :].sum(
            dim=1)  # Shape: [batch_size, sequence_length]
        relevance_scores += cls_attention  # Accumulate across layers

    # Average across the batch if you have multiple samples and want a single relevance score per token
    avg_relevance_scores = relevance_scores.mean(
        dim=0).detach().numpy()  # Shape: [sequence_length]

    # normalize the scores
    normalized_scores = avg_relevance_scores / avg_relevance_scores.sum()

    normalized_scores[0] = 0
    sum_normalized_scores = normalized_scores.sum()
    new_normalized_scores = normalized_scores / sum_normalized_scores

    # disregard the first index

    return new_normalized_scores


# matplotlib.use('TkAgg')
# Load the dataset configuration
data_config_path = "config/data_config.yaml"
with open(data_config_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Set up the dataset and dataloader
dataset = UAVTDataset(config=data_config, is_validation=False)
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, collate_fn=dataset.collate_fn)

# Load the latest checkpoint
# checkpoint_dir = "evader_former_checkpoint/"
checkpoint_dir = "uavt_checkpoint/"
checkpoint_files = sorted(
    [os.path.join(checkpoint_dir, f)
     for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
    key=os.path.getmtime
)

model_config = {}
# Ensure there is a checkpoint to load
if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading model from checkpoint: {latest_checkpoint}")

    # Load the model directly from the checkpoint using the class method
    model = EvaderFormer.load_from_checkpoint(
        latest_checkpoint, config=model_config)
else:
    raise FileNotFoundError("No checkpoint found in the directory.")

# Set up the device
device = torch.device("cpu")
model.to(device)  # Move the model to the appropriate device
model.eval()  # Set the model to evaluation mode

# Run inference on a batch
with torch.no_grad():  # Disable gradient calculation for inference
    for i, batch in enumerate(dataloader):
        # idx, target, waypoints = batch['input'], batch['output'], batch['waypoints']
        # print("Predicted Waypoints:", predicted_waypoints)
        # print("Attention Map:", attn_map)  # This is the attention values
        # break  # Exit after the first batch for demonstration purposes
        if i == 1:
            break

        for k, v in batch.items():
            # load to the same device as the model
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)

        # Get the model's output and attention values
        _, predicted_waypoints, attn_map = model(
            batch)

        # plot the predicted waypoints
        # Extract the predicted waypoints
        # for i in range(1):
        #     fig, ax = plt.subplots()
        #     predicted_waypoints = predicted_waypoints
        #     ax.plot(predicted_waypoints[i, :, 0],
        #             predicted_waypoints[i, :, 1], label='Predicted Waypoints')
        #     # plot the waypoints
        #     waypoints = batch['waypoints'].cpu().numpy()
        #     ax.plot(waypoints[i, :, 0], waypoints[i, :, 1], label='Waypoints')

        #     ax.set_title(f"Predicted Waypoints for Sample {i+1}")
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.legend()

        # Sum over all layers and heads for the [CLS] token's attention
        # Initialize an array to accumulate attention scores
        batch_size, num_tokens = attn_map[0].shape[0], attn_map[0].shape[-1]
        # Shape: [batch_size, sequence_length]
        relevance_scores = torch.zeros(batch_size, num_tokens)

        # Sum attention weights across all layers and heads
        for layer_attention in attn_map:
            # Sum over heads to get the attention distribution of the [CLS] token across the sequence
            cls_attention = layer_attention[:, :, 0, :].sum(
                dim=1)  # Shape: [batch_size, sequence_length]
            relevance_scores += cls_attention  # Accumulate across layers

        # Average across the batch if you have multiple samples and want a single relevance score per token
        avg_relevance_scores = relevance_scores.mean(
            dim=0).cpu().numpy()  # Shape: [sequence_length]

        # normalize the scores
        normalized_scores = avg_relevance_scores / avg_relevance_scores.sum()

        pursuer_indices = [1, 2, 3]
        # Extract relevance scores for pursuers
        # Shape: (number of pursuers,)
        pursuer_relevance_scores = normalized_scores[pursuer_indices]
        # print(pursuer_relevance_scores)
        # # Plot the relevance scores for pursuers
        # # plt.figure(figsize=(10, 6))
        # fig, ax = plt.subplots(figsize=(10, 6))

        # plt.bar(range(len(pursuer_indices)), pursuer_relevance_scores, tick_label=[
        #         f"Pursuer {i+1}" for i in range(len(pursuer_indices))])
        # plt.title(
        #     "Relevance Scores for Pursuers (Summed Across All Layers and Heads)")
        # plt.xlabel("Pursuer")
        # plt.ylabel("Relevance Score")


"""
This is an example if we were to deploy 
the model and use it to predict the waypoints of the ego vehicle.
We need to input the data where it's normalized or centered around 
the ego vehicle.
Once we get the prediction we need to map it 
back to the global coordinates.
"""

# get first 150 samples
# samples = dataloader.dataset[:250]
data_info = dataset.data_info
# get all the values of the first key
# get the first key
keys = list(data_info.keys())
# get all the values of the first key
# 30 is FUCKING WILD
samples = data_info[keys[30]] # this is 30
ego = AgentHistory()
pursuer_1 = AgentHistory()
pursuer_2 = AgentHistory()
pursuer_3 = AgentHistory()
inference_time: List[float] = []

# To see what is going on we need to map it back to global position
first_sample = samples[10]
sample_json: str = first_sample['filename']
json_data = JSONData(sample_json)

fill_dummy: bool = False

for i, s in enumerate(samples):
    batch: Dict[str, Any] = dataloader.dataset.collate_fn([s])
    """
    """
    # an idiot check to make sure attention value is doing something
    if fill_dummy:
        batch['input'][0][0, 2] = 500
        batch['input'][0][0, 3] = 500

    # # pickle the batch
    # import pickle
    # if i == 0:
    #     with open('batch.pkl', 'wb') as f:
    #         pickle.dump(batch, f)

    start_time = time.time()
    _, predicted_waypoints, attn_map = model(
        batch)
    final_time = time.time()
    inference_time.append(final_time - start_time)
    # print(f"Time taken for inference: {time.time() - start_time:.2f} seconds")
    norm_attention_scores = compute_attention_scores(attn_map)
    pursuer_relevance_scores = norm_attention_scores[pursuer_indices]

    # because the waypoints are in relative coordinates, we need to map them back to global coordinates
    predicted_waypoints = predicted_waypoints.detach().numpy().squeeze()
    bias_position = batch['bias_position'].detach().numpy().squeeze()

    # get the ego vehicle's position
    global_predicted_waypoints = predicted_waypoints + bias_position[0:3]
    waypoints = batch['waypoints'].detach(
    ).numpy().squeeze() + bias_position[0:3]

    # store the ego vehicle's information
    ego.x.append(bias_position[0])
    ego.y.append(bias_position[1])
    ego.z.append(bias_position[2])
    ego.psi.append(bias_position[3])
    ego.waypoints_x.extend(waypoints[:, 0])
    ego.waypoints_y.extend(waypoints[:, 1])
    ego.pred_waypoints_x.extend(global_predicted_waypoints[:, 0])
    ego.pred_waypoints_y.extend(global_predicted_waypoints[:, 1])

    # store the pursuer's information which is stored in the input
    # each row consists of a pursuer's information
    pursuers = list(batch['input'][0].detach().numpy())
    for i, p in enumerate(pursuers):
        # we need to unbias the pursuer's position
        x_idx: int = 2
        y_idx: int = 3
        z_idx: int = 4
        p[x_idx] += bias_position[0]
        p[y_idx] += bias_position[1]
        p[z_idx] += bias_position[2]

        if i == 0:
            pursuer_1.x.append(p[x_idx])
            pursuer_1.y.append(p[y_idx])
            pursuer_1.z.append(p[z_idx])
            pursuer_1.attention_scores.append(pursuer_relevance_scores[0])
        elif i == 1:
            pursuer_2.x.append(p[x_idx])
            pursuer_2.y.append(p[y_idx])
            pursuer_2.z.append(p[z_idx])
            pursuer_2.attention_scores.append(pursuer_relevance_scores[1])

        else:
            pursuer_3.x.append(p[x_idx])
            pursuer_3.y.append(p[y_idx])
            pursuer_3.z.append(p[z_idx])
            pursuer_3.attention_scores.append(pursuer_relevance_scores[2])

pursuer_list = [pursuer_1, pursuer_2, pursuer_3]
pursuer_colors = ['red', 'orange', 'green']
fig, ax = plt.subplots()
ax.plot(ego.x, ego.y, label='Ego', color='black')
ax.scatter(ego.x[0], ego.y[0], color='black', marker='x', label='Start')
ax.plot(ego.pred_waypoints_x, ego.pred_waypoints_y,
        label='Predicted Waypoints')
for i, pursuer in enumerate(pursuer_list):
    ax.plot(pursuer.x, pursuer.y, label=f'Pursuer {i+1}',
            color=pursuer_colors[i])
    ax.scatter(pursuer.x[0], pursuer.y[0],
               color=pursuer_colors[i], marker='x', label='Start')
ax.legend()
ax.set_title("Ego and Pursuers Trajectory")

fig, ax = plt.subplots()
for i, p in enumerate(pursuer_list):
    ax.plot(p.attention_scores, label=f"Pursuer {i+1}",
            color=pursuer_colors[i])
ax.legend()
ax.set_title("Attention Scores for Pursuers")

# Let's look at the error between the predicted waypoints and the actual waypoints
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
# ax[0].plot(ego.pred_waypoints_x, label='Predicted Waypoints X')
# ax[0].plot(ego.waypoints_x, label='Actual Waypoints X')
error_x = np.abs(np.array(ego.pred_waypoints_x) -
                 np.array(ego.waypoints_x))
mse_x = np.mean(error_x)
ax[0].plot(error_x, label='Error X')
ax[0].set_title(f"X Coordinates, MSE: {mse_x:.2f}")

# ax[1].plot(ego.pred_waypoints_y, label='Predicted Waypoints Y')
# ax[1].plot(ego.waypoints_y, label='Actual Waypoints Y')
error_y = np.abs(np.array(ego.pred_waypoints_y) -
                 np.array(ego.waypoints_y))
mse_y = np.mean(error_y)
ax[1].plot(error_y, label='Error Y')
ax[1].set_title(f"Y Coordinates, MSE: {mse_y:.2f}")

# ax[2].plot(ego.pred_waypoints_x, ego.pred_waypoints_y,
#            label='Predicted Waypoints')
# ax[2].plot(ego.waypoints_x, ego.waypoints_y, label='Actual Waypoints')
# ax[2].plot(ego.pred_waypoints_x, ego.pred_waypoints_y,
#            label='Predicted Waypoints')
# ax[2].set_title("Predicted vs Actual Waypoints")
error_z = np.abs(np.array(ego.pred_waypoints_x) -
                 np.array(ego.waypoints_x))
mse_z = np.mean(error_z)
ax[2].plot(error_z, label='Error Z')
ax[2].set_title(f"Z Coordinates, MSE: {mse_z:.2f}")

for a in ax:
    a.legend()


# plot the json data
# fig, ax = plt.subplots()
# ax.plot(json_data.ego.x, json_data.ego.y, label='Ego', color='black')
# ax.scatter(json_data.ego.x[0], json_data.ego.y[0],
#            color='black', marker='x', label='Start')
# ax.plot(json_data.pursuer_1.x, json_data.pursuer_1.y,
#         label='Pursuer 1', color='red')
# ax.scatter(json_data.pursuer_1.x[0], json_data.pursuer_1.y[0],
#            color='red', marker='x', label='Start')
# %%
# ax.plot(json_data.pursuer_2.x, json_data.pursuer_2.y,

#         label='Pursuer 2', color='orange')
# ax.scatter(json_data.pursuer_2.x[0], json_data.pursuer_2.y[0],
#            color='orange', marker='x', label='Start')
# ax.legend()
# ax.set_title("JSON Data Trajectory")

# # an idiot check to make sure the data is the same
# fig, ax = plt.subplots()
# ax.plot(json_data.ego.x, json_data.ego.y, label='Ego', color='black')
# ax.scatter(json_data.ego.x[0], json_data.ego.y[0],
#            color='black', marker='x', label='Start')
# ax.plot(ego.x, ego.y, label='Ego', color='blue')
# ax.scatter(ego.x[0], ego.y[0], color='blue', marker='x', label='Start')
# ax.legend()
# ax.set_title("Ego Trajectory Comparison")

# # an idiot check to make sure the pursuer data is the same
# fig, ax = plt.subplots()
# ax.plot(json_data.pursuer_1.x, json_data.pursuer_1.y,
#         label='Pursuer 1 JSON', color='red')
# ax.scatter(json_data.pursuer_1.x[0], json_data.pursuer_1.y[0],
#            color='red', marker='x', label='Start')
# ax.plot(pursuer_1.x, pursuer_1.y, label='Pursuer 1', color='blue')
# ax.scatter(pursuer_1.x[0], pursuer_1.y[0],
#            color='blue', marker='x', label='Start')

# ax.plot(json_data.pursuer_2.x, json_data.pursuer_2.y,
#         label='Pursuer 2 JSON', color='orange')
# ax.scatter(json_data.pursuer_2.x[0], json_data.pursuer_2.y[0],
#            color='orange', marker='x', label='Start')
# ax.plot(pursuer_2.x, pursuer_2.y, label='Pursuer 2', color='green')
# ax.scatter(pursuer_2.x[0], pursuer_2.y[0],
#            color='green', marker='x', label='Start')
# ax.legend()
# ax.set_title("Pursuer 1 Trajectory Comparison")

# fig, ax = plt.subplots()
# ax.plot(inference_time)
# ax.set_title("Inference Time")

# # animate the ego vehicle and pursuers
# fig, ax = plt.subplots()
# # Set up color normalization and colormap
# # Adjust based on min and max of attention scores
# norm = Normalize(vmin=0, vmax=0.6)
# # Use a colormap that visually differentiates scores
# cmap = plt.get_cmap("magma")

# ego_line, = ax.plot([], [], label='Ego', color='blue')
# pursuer_1_line, = ax.plot([], [], label='Pursuer 1', color='red')
# pursuer_2_line, = ax.plot([], [], label='Pursuer 2', color='orange')
# pursuer_3_line, = ax.plot([], [], label='Pursuer 3', color='green')

# # Color bar setup
# sm = ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])  # Dummy array for color bar
# # Add color bar with label
# cbar = fig.colorbar(sm, ax=ax, label="Attention Score")
# min_x = min(ego.x)
# max_x = max(ego.x)
# min_y = min(ego.y)
# max_y = max(ego.y)

# pursuer_1 = pursuer_list[0]
# pursuer_2 = pursuer_list[1]
# pursuer_3 = pursuer_list[2]


# def init():
#     ax.set_xlim(-800, 400)
#     ax.set_ylim(-800, 400)

#     return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line


# # Update function for animation
# def update(frame):
#     # Ego path update
#     ego_line.set_data(ego.x[:frame], ego.y[:frame])

#     # Pursuer paths update
#     pursuer_1_line.set_data(pursuer_1.x[:frame], pursuer_1.y[:frame])
#     pursuer_2_line.set_data(pursuer_2.x[:frame], pursuer_2.y[:frame])
#     pursuer_3_line.set_data(pursuer_3.x[:frame], pursuer_3.y[:frame])
#     # # Map the current attention score to color for each pursuer
#     pursuer_1_color = cmap(norm(pursuer_1.attention_scores[frame]))
#     pursuer_2_color = cmap(norm(pursuer_2.attention_scores[frame]))
#     pursuer_3_color = cmap(norm(pursuer_3.attention_scores[frame]))
#     # # Update line colors based on attention scores
#     pursuer_1_line.set_color(pursuer_1_color)
#     pursuer_2_line.set_color(pursuer_2_color)
#     pursuer_3_line.set_color(pursuer_3_color)

#     # for i, p in enumerate(pursuer_list):
#     #     pursuer_lines[i][0].set_data(p.x[:frame], p.y[:frame])
#     #     pursuer_color = cmap(norm(p.attention_scores[frame]))
#     #     pursuer_lines[i][0].set_color(pursuer_color)

#     return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line


# ax.legend()
# ani = animation.FuncAnimation(fig, update, frames=len(ego.x),
#                               init_func=init, blit=True, interval=20)

# plt.show()

# Set up the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up color normalization and colormap
norm = Normalize(vmin=0, vmax=0.6)
cmap = plt.get_cmap("plasma")

# Initialize lines for ego and pursuers
ego_line, = ax.plot([], [], [], label='Ego', color='blue')
pursuer_1_line, = ax.plot([], [], [], label='Pursuer 1',
                          color='red', linestyle='--')  # Dashed line
pursuer_2_line, = ax.plot([], [], [], label='Pursuer 2',
                          color='orange', linestyle=':')  # Dotted line
pursuer_3_line, = ax.plot([], [], [], label='Pursuer 3',
                          color='green', linestyle='-.')  # Dash-dot line

# Color bar setup
# Color bar setup
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Dummy array for color bar
cbar = fig.colorbar(sm, ax=ax, label="Attention Score")

# Set limits for the 3D plot
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_zlim(-30, 30)  # Adjust Z limits based on your data

# Set labels for axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


def init():
    ego_line.set_data([], [])
    ego_line.set_3d_properties([])
    pursuer_1_line.set_data([], [])
    pursuer_1_line.set_3d_properties([])
    pursuer_2_line.set_data([], [])
    pursuer_2_line.set_3d_properties([])
    pursuer_3_line.set_data([], [])
    pursuer_3_line.set_3d_properties([])
    return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line

# Update function for the animation


def update(frame):
    # Ego vehicle update
    ego_line.set_data(ego.x[:frame], ego.y[:frame])
    ego_line.set_3d_properties(ego.z[:frame])

    # Pursuer updates
    pursuer_1_line.set_data(pursuer_1.x[:frame], pursuer_1.y[:frame])
    pursuer_1_line.set_3d_properties(pursuer_1.z[:frame])

    pursuer_2_line.set_data(pursuer_2.x[:frame], pursuer_2.y[:frame])
    pursuer_2_line.set_3d_properties(pursuer_2.z[:frame])

    pursuer_3_line.set_data(pursuer_3.x[:frame], pursuer_3.y[:frame])
    pursuer_3_line.set_3d_properties(pursuer_3.z[:frame])

    # Update line colors based on attention scores
    pursuer_1_color = cmap(norm(pursuer_1.attention_scores[frame]))
    pursuer_2_color = cmap(norm(pursuer_2.attention_scores[frame]))
    pursuer_3_color = cmap(norm(pursuer_3.attention_scores[frame]))
    pursuer_1_line.set_color(pursuer_1_color)
    pursuer_2_line.set_color(pursuer_2_color)
    pursuer_3_line.set_color(pursuer_3_color)

    return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(ego.x),
                              init_func=init, blit=True, interval=40)

ax.legend()
plt.show()
