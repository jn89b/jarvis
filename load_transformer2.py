import seaborn as sns
import random
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
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from jarvis.transformers.evaderformer2 import EvaderFormer
from jarvis.datasets.base_dataset import PlanTDataset, UAVTDataset
plt.close('all')


# Function to compute correlation
def compute_correlation(attribute, attention_scores):
    if len(attribute) > 1 and len(attention_scores) > 1:  # Ensure valid data
        # Pearson correlation
        return np.corrcoef(attribute, attention_scores)[0, 1]
    else:
        return None  # Not enough data to compute correlation


class AgentHistory():
    def __init__(self) -> None:
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self.psi: List[float] = []
        self.v: List[float] = []
        self.vx: List[float] = []
        self.vy: List[float] = []
        self.vz: List[float] = []
        self.waypoints_x: List[float] = []
        self.waypoints_y: List[float] = []
        self.waypoints_z: List[float] = []
        self.pred_waypoints_x: List[float] = []
        self.pred_waypoints_y: List[float] = []
        self.pred_waypoints_z: List[float] = []
        self.attention_scores: List[float] = []
        self.dr: List[float] = []  # distance to reference
        self.dv: List[float] = []  # velocity difference
        self.ttcs: List[float] = []  # time to collision
        self.delta_ttcs: List[float] = []  # change in time to collision


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
    relevance_scores = torch.zeros(batch_size, num_tokens-1)
    min_relevance_scores = None  # Initialize for tracking minimum scores
    min_cls_value = 1000  # Initialize with a small value
    desired_layer: int = 0
    # Sum attention weights across all layers and heads
    for i, layer_attention in enumerate(attention_map):
        # Sum over heads to get the attention distribution of the [CLS] token across the sequence
        cls_attention = layer_attention[:, :, 0, 1:].sum(
            dim=1)  # Shape: [batch_size, sequence_length]
        relevance_scores += cls_attention  # Accumulate across layers
        # Compute the minimum scores
        cls_value = cls_attention[0][0]
        if cls_value < min_cls_value:
            min_cls_value = cls_value
            desired_layer = i
            min_relevance_scores = cls_attention

    # min_relevance_scores = min_relevance_scores.detach().numpy()
    # Average across the batch if you have multiple samples and want a single relevance score per token
    avg_relevance_scores = relevance_scores.mean(
        dim=0).detach().numpy()  # Shape: [sequence_length]
    # normalize the scores
    normalized_scores = avg_relevance_scores / avg_relevance_scores.sum()
    # normalized_scores[0] = 0
    sum_normalized_scores = normalized_scores.sum()
    new_normalized_scores = normalized_scores / sum_normalized_scores

    # disregard the first index
    min_relevance_scores = min_relevance_scores.detach().numpy().flatten()
    #  set the shape
    # print("min relevance scores: ", min_relevance_scores)
    # min_relevance_scores[0] = 0
    sum_min_relevance_scores = min_relevance_scores.sum()
    # print("sum_min_relevance_scores: ", sum_min_relevance_scores)
    min_relevance_scores = min_relevance_scores / sum_min_relevance_scores

    return min_relevance_scores


# matplotlib.use('TkAgg')
# Load the dataset configuration
data_config_path = "config/data_config.yaml"
with open(data_config_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Set up the dataset and dataloader
dataset = UAVTDataset(config=data_config, is_validation=True)
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

        pursuer_indices = [0, 1, 2]
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
# random number
random_key = random.randint(0, len(keys))
# get all the values of the first key
# 30 is FUCKING WILD
print("random key: ", random_key)
samples = data_info[keys[random_key]]  # this is 30
ego = AgentHistory()
pursuer_1 = AgentHistory()
pursuer_2 = AgentHistory()
pursuer_3 = AgentHistory()
inference_time: List[float] = []

# To see what is going on we need to map it back to global position
first_sample = samples[5]
sample_json: str = first_sample['filename']
json_data = JSONData(sample_json)

fill_dummy: bool = False

for i, s in enumerate(samples):
    batch: Dict[str, Any] = dataloader.dataset.collate_fn([s])
    """
    """

    # if i == 5:
    #     break

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
    ego.waypoints_z.extend(waypoints[:, 2])
    ego.pred_waypoints_x.extend(global_predicted_waypoints[:, 0])
    ego.pred_waypoints_y.extend(global_predicted_waypoints[:, 1])
    ego.pred_waypoints_z.extend(global_predicted_waypoints[:, 2])
    # an idiot check to make sure attention value is doing something
    if fill_dummy:
        # convert to torch tensor
        batch['input'][0][2, 2] = torch.tensor(predicted_waypoints[0, 0]-15)
        batch['input'][0][2, 3] = torch.tensor(predicted_waypoints[0, 1]+15)
        batch['input'][0][2, 4] = torch.tensor(predicted_waypoints[0, 2])

        # batch['input'][0][1, 2] = torch.tensor(predicted_waypoints[0, 0]+30)
        # batch['input'][0][1, 3] = torch.tensor(predicted_waypoints[0, 1]-50)
        # batch['input'][0][1, 4] = torch.tensor(predicted_waypoints[0, 2])

        # batch['input'][0][1, 2] = torch.tensor(bias_position[0])
        # batch['input'][0][1, 3] = torch.tensor(bias_position[1])

    # store the pursuer's information which is stored in the input
    # each row consists of a pursuer's information
    pursuers = list(batch['input'][0].detach().numpy())
    for i, p in enumerate(pursuers):
        # we need to unbias the pursuer's position
        x_idx: int = 2
        y_idx: int = 3
        z_idx: int = 4
        phi_idx: int = 5
        theta_idx: int = 6
        psi_idx: int = 7
        dv_idx: int = 8
        p[x_idx] += bias_position[0]
        p[y_idx] += bias_position[1]
        p[z_idx] += bias_position[2]
        psi = p[psi_idx]
        dr = np.sqrt((p[x_idx] - ego.x[-1])**2 +
                     (p[y_idx] - ego.y[-1])**2 +
                     (p[z_idx] - ego.z[-1])**2)

        # let's see if we can visualize some kind of correlation value from this??

        if i == 0:
            pursuer_1.x.append(p[x_idx])
            pursuer_1.y.append(p[y_idx])
            pursuer_1.z.append(p[z_idx])
            pursuer_1.v.append(p[dv_idx])
            # dv = np.abs(p[5] - ego.v[-1])
            pursuer_1.dr.append(dr)
            pursuer_1.psi.append(psi)
            # pursuer_1.dv.append(dv)
            pursuer_1.attention_scores.append(pursuer_relevance_scores[0])
        elif i == 1:
            pursuer_2.x.append(p[x_idx])
            pursuer_2.y.append(p[y_idx])
            pursuer_2.z.append(p[z_idx])
            pursuer_2.v.append(p[dv_idx])
            # dv = np.abs(p[5] - ego.v[-1])
            pursuer_2.dr.append(dr)
            pursuer_2.psi.append(psi)
            # pursuer_2.dv.append(dv)
            pursuer_2.attention_scores.append(pursuer_relevance_scores[1])

        else:
            pursuer_3.x.append(p[x_idx])
            pursuer_3.y.append(p[y_idx])
            pursuer_3.z.append(p[z_idx])
            # dv = np.abs(p[5] - ego.v[-1])
            pursuer_3.v.append(p[dv_idx])
            pursuer_3.dr.append(dr)
            pursuer_3.psi.append(psi)
            # pursuer_3.dv.append(dv)
            pursuer_3.attention_scores.append(pursuer_relevance_scores[2])


# %%
save_folder: str = "figures"
matplotlib.rc('font', size=16)          # Base font
matplotlib.rc('axes', labelsize=18)     # X, Y labels
matplotlib.rc('axes', titlesize=22)     # Axes title
matplotlib.rc('xtick', labelsize=16)    # Tick labels
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('figure', titlesize=24)   # Figure "suptitle"
plt.close('all')
sns.set_palette("colorblind")
time_overall = 30.0
time_vector = np.linspace(0, time_overall, len(ego.waypoints_x))

pursuer_1_dr_corr = compute_correlation(
    pursuer_1.dr, pursuer_1.attention_scores)
pursuer_1_psi_corr = compute_correlation(
    pursuer_1.psi, pursuer_1.attention_scores)

pursuer_2_dr_corr = compute_correlation(
    attribute=pursuer_2.dr, attention_scores=pursuer_2.attention_scores)
pursuer_2_psi_corr = compute_correlation(
    attribute=pursuer_2.psi, attention_scores=pursuer_2.attention_scores)

pursuer_list = [pursuer_1, pursuer_2, pursuer_3]

# compute the diff of x,y,z to get velocity
time_step: float = 0.1
for p in pursuer_list:
    p.vx = np.diff(p.x)/time_step
    p.vy = np.diff(p.y)/time_step
    p.vz = np.diff(p.z)/time_step
ego.vx = np.diff(ego.x)/time_step
ego.vy = np.diff(ego.y)/time_step
ego.vz = np.diff(ego.z)/time_step

# compute the time to collision
for p in pursuer_list:
    # pursuer x and y are already relative where ego is centered
    # we will add a negative to sign to flip the convention
    dx = np.array(ego.x[1:]) - np.array(p.x[1:])
    dy = np.array(ego.y[1:]) - np.array(p.y[1:])
    dz = np.array(ego.z[1:]) - np.array(p.z[1:])
    dvx = ego.vx - p.vx
    dvy = ego.vy - p.vy
    dvz = ego.vz - p.vz
    # compute the time to collision
    p.ttcs = -(dx*dvx + dy*dvy + dz*dvz) / \
        (dvx**2 + dvy**2 + dvz**2)
    p.delta_ttcs = np.diff(p.ttcs) / time_step

pursuer_colors = ['red', 'orange', 'green']
fig, ax = plt.subplots()
ax.plot(ego.x, ego.y, label='Ego', color='black')
ax.scatter(ego.x[0], ego.y[0], color='black', marker='x', label='Start')
ax.scatter(ego.pred_waypoints_x, ego.pred_waypoints_y,
           label='Predicted Waypoints', color='orange')
for i, pursuer in enumerate(pursuer_list):
    ax.plot(pursuer.x, pursuer.y, label=f'Pursuer {i+1}',
            color=pursuer_colors[i])
    ax.scatter(pursuer.x[0], pursuer.y[0],
               color=pursuer_colors[i], marker='x', label='Start')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.set_title("Ego and Pursuers Trajectory")

# fig, ax = plt.subplots()
fig, ax = plt.subplots(5, 1, figsize=(10, 12))
for i, p in enumerate(pursuer_list):
    ax[0].plot(p.attention_scores, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[0].set_title("Attention Scores for Pursuers")
    ax[1].plot(p.dr, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[1].set_title("Distance to Ego for Pursuers")
    ax[2].plot(p.psi, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[2].set_title("Psi for Pursuers")
    ax[3].plot(p.ttcs, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[3].set_title("Time to Collision for Pursuers")
    ax[4].plot(p.delta_ttcs, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[4].set_title("Change in Time to Collision for Pursuers")
for a in ax:
    a.legend()

# set supertitle
fig.suptitle("Attention Scores and Distance to Ego for Pursuers")
# ax.set_title("Attention Scores WRT to Distance from Ego for Pursuers")

fig, ax = plt.subplots(4, 1, figsize=(10, 12))
for i, p in enumerate(pursuer_list):
    ax[0].plot(p.v, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[0].set_title("Velocity for Pursuers")
    ax[1].plot(p.vx, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[1].set_title("X Velocity for Pursuers")
    ax[2].plot(p.vy, label=f"Pursuer {i+1}",
               color=pursuer_colors[i])
    ax[2].set_title("Y Velocity for Pursuers")
    estimated_v = np.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
    ax[3].plot(estimated_v, label=f"Estimated Pursuer {i+1}",
               color='blue')
for a in ax:
    a.legend()


# Let's look at the error between the predicted waypoints and the actual waypoints
fig, ax = plt.subplots(3, 1, figsize=(10, 12))
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

error_z = np.abs(np.array(ego.pred_waypoints_x) -
                 np.array(ego.waypoints_x))
mse_z = np.mean(error_z)
ax[2].plot(error_z, label='Error Z')
ax[2].set_title(f"Z Coordinates, MSE: {mse_z:.2f}")

for a in ax:
    a.legend()

# Let's plot the x prediction vs the actual x
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# set a supertitle with the mse for each coordinate
fig.suptitle(f"MSE for X: {mse_x:.2f}, Y: {mse_y:.2f}, Z: {mse_z:.2f}")
ax[0].scatter(time_vector, ego.pred_waypoints_x,
              label='Predicted Waypoints X', color='orange')
ax[0].plot(time_vector, ego.waypoints_x, label='Actual Waypoints X')

ax[1].scatter(time_vector, ego.pred_waypoints_y,
              label='Predicted Waypoints Y', color='orange')
ax[1].plot(time_vector, ego.waypoints_y, label='Actual Waypoints Y')

ax[2].scatter(time_vector, ego.pred_waypoints_z,
              label='Predicted Waypoints Z', color='orange')
ax[2].plot(time_vector, ego.waypoints_z, label='Actual Waypoints Z')

for a in ax:
    a.legend()
# share the x axiss
# tight axis
fig.tight_layout()
# save figure
fig.savefig(save_folder+"/predicted_waypoints.svg")
# plot the json data
# fig, ax = plt.subplots()
# ax.plot(json_data.ego.x, json_data.ego.y, label='Ego', color='black')
# ax.plot(json_data.ego.x[0], json_data.ego.y[0],
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

# # Set up the 3D figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Set up color normalization and colormap
# norm = Normalize(vmin=0, vmax=0.6)
# cmap = plt.get_cmap("plasma")

# # Initialize lines for ego and pursuers
# ego_line, = ax.plot([], [], [], label='Ego', color='blue')
# pursuer_1_line, = ax.plot([], [], [], label='Pursuer 1',
#                           color='red', linestyle='--')  # Dashed line
# pursuer_2_line, = ax.plot([], [], [], label='Pursuer 2',
#                           color='orange', linestyle=':')  # Dotted line
# pursuer_3_line, = ax.plot([], [], [], label='Pursuer 3',
#                           color='green', linestyle='-.')  # Dash-dot line

# # Color bar setup
# # Color bar setup
# sm = ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])  # Dummy array for color bar
# cbar = fig.colorbar(sm, ax=ax, label="Attention Score")

# # Set limits for the 3D plot
# ax.set_xlim(-200, 400)
# ax.set_ylim(-200, 400)
# ax.set_zlim(-30, 30)  # Adjust Z limits based on your data

# # Set labels for axes
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# def init():
#     ego_line.set_data([], [])
#     ego_line.set_3d_properties([])
#     pursuer_1_line.set_data([], [])
#     pursuer_1_line.set_3d_properties([])
#     pursuer_2_line.set_data([], [])
#     pursuer_2_line.set_3d_properties([])
#     pursuer_3_line.set_data([], [])
#     pursuer_3_line.set_3d_properties([])
#     return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line

# # Update function for the animation

# def update(frame):
#     # show only the last 20 frames
#     last_frame = 75
#     if frame < last_frame:
#         start_frame = 0
#     else:
#         start_frame = frame - last_frame
#     # Ego vehicle update
#     ego_line.set_data(ego.x[start_frame:frame], ego.y[start_frame:frame])
#     ego_line.set_3d_properties(ego.z[start_frame:frame])

#     # Pursuer updates
#     frame_span = slice(start_frame, frame)
#     pursuer_1_line.set_data(pursuer_1.x[frame_span], pursuer_1.y[frame_span])
#     pursuer_1_line.set_3d_properties(pursuer_1.z[frame_span])

#     pursuer_2_line.set_data(pursuer_2.x[frame_span], pursuer_2.y[frame_span])
#     pursuer_2_line.set_3d_properties(pursuer_2.z[frame_span])

#     pursuer_3_line.set_data(pursuer_3.x[frame_span], pursuer_3.y[frame_span])
#     pursuer_3_line.set_3d_properties(pursuer_3.z[frame_span])

#     # Update line colors based on attention scores
#     pursuer_1_color = cmap(norm(pursuer_1.attention_scores[frame]))
#     pursuer_2_color = cmap(norm(pursuer_2.attention_scores[frame]))
#     pursuer_3_color = cmap(norm(pursuer_3.attention_scores[frame]))
#     pursuer_1_line.set_color(pursuer_1_color)
#     pursuer_2_line.set_color(pursuer_2_color)
#     pursuer_3_line.set_color(pursuer_3_color)

#     return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line


# # Create animation
# ani = animation.FuncAnimation(fig, update, frames=len(ego.x),
#                               init_func=init, blit=True, interval=40)

# ax.legend()
# plt.show()
# Set up the figure with a 3D subplot and a bar chart subplot
# make a subplot with 1 row and 2 columns


# Define a variant red palette for the pursuers
pursuer_colors = ['#ff9999', '#ff4d4d', '#cc0000']  # light, medium, dark red

# First figure: Distance vs. Attention Score
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for i, p in enumerate(pursuer_list):
    # Plot distance vs. attention score for each pursuer
    ax.plot(p.attention_scores, p.dr, label=f"Pursuer {i+1}",
            color=pursuer_colors[i])
ax.legend()

# Second figure: 3D plot and bar chart
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])  # 2 columns: bar chart and 3D plot

# Bar chart on the left
ax_bar = fig.add_subplot(gs[0])
ax_bar.set_ylim(0, 0.6)  # Attention score range
bar_labels = ['Pursuer 1', 'Pursuer 2', 'Pursuer 3']
bars = ax_bar.bar(bar_labels, [0, 0, 0], color=pursuer_colors)
ax_bar.set_ylabel('Attention Score')
ax_bar.set_title('Attention Values')

# 3D plot on the right
ax_3d = fig.add_subplot(gs[1], projection='3d')

# Set up color normalization and colormap (if needed later)
norm = Normalize(vmin=0, vmax=0.6)
cmap = plt.get_cmap("plasma")

# Initialize lines for ego and pursuers in the 3D plot
ego_line, = ax_3d.plot([], [], [], label='Ego', color='blue')
pursuer_1_line, = ax_3d.plot(
    [], [], [], label='Pursuer 1', color=pursuer_colors[0])
pursuer_2_line, = ax_3d.plot(
    [], [], [], label='Pursuer 2', color=pursuer_colors[1])
pursuer_3_line, = ax_3d.plot(
    [], [], [], label='Pursuer 3', color=pursuer_colors[2])

# Instead of hard-coded axis limits, compute limits based on the data.
# This assumes ego.x, pursuer_1.x, etc. have been filled with trajectory data.
all_x = ego.x + pursuer_1.x + pursuer_2.x + pursuer_3.x
all_y = ego.y + pursuer_1.y + pursuer_2.y + pursuer_3.y
all_z = ego.z + pursuer_1.z + pursuer_2.z + pursuer_3.z

x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
z_min, z_max = min(all_z), max(all_z)

# Add a 10% margin to the computed bounds
margin_x = 0.1 * (x_max - x_min)
margin_y = 0.1 * (y_max - y_min)
margin_z = 0.1 * (z_max - z_min)

ax_3d.set_xlim(x_min - margin_x, x_max + margin_x)
ax_3d.set_ylim(y_min - margin_y, y_max + margin_y)
ax_3d.set_zlim(z_min - margin_z, z_max + margin_z)

# Set labels for axes
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")


def init():
    ego_line.set_data([], [])
    ego_line.set_3d_properties([])
    pursuer_1_line.set_data([], [])
    pursuer_1_line.set_3d_properties([])
    pursuer_2_line.set_data([], [])
    pursuer_2_line.set_3d_properties([])
    pursuer_3_line.set_data([], [])
    pursuer_3_line.set_3d_properties([])
    for bar in bars:
        bar.set_height(0)  # Initialize bar heights to zero
    return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line, *bars


def update(frame):
    # Define the frame range for trailing effect
    last_frame = 75
    start_frame = max(0, frame - last_frame)
    frame_span = slice(start_frame, frame)

    # Update 3D trajectories for ego and pursuers
    ego_line.set_data(ego.x[frame_span], ego.y[frame_span])
    ego_line.set_3d_properties(ego.z[frame_span])

    pursuer_1_line.set_data(pursuer_1.x[frame_span], pursuer_1.y[frame_span])
    pursuer_1_line.set_3d_properties(pursuer_1.z[frame_span])

    pursuer_2_line.set_data(pursuer_2.x[frame_span], pursuer_2.y[frame_span])
    pursuer_2_line.set_3d_properties(pursuer_2.z[frame_span])

    pursuer_3_line.set_data(pursuer_3.x[frame_span], pursuer_3.y[frame_span])
    pursuer_3_line.set_3d_properties(pursuer_3.z[frame_span])

    # Ensure the pursuer line colors remain our variant red palette
    pursuer_1_line.set_color(pursuer_colors[0])
    pursuer_2_line.set_color(pursuer_colors[1])
    pursuer_3_line.set_color(pursuer_colors[2])

    # Update bar chart heights based on attention scores
    bar_heights = [
        pursuer_1.attention_scores[frame],
        pursuer_2.attention_scores[frame],
        pursuer_3.attention_scores[frame],
    ]
    for bar, height in zip(bars, bar_heights):
        bar.set_height(height)

    return ego_line, pursuer_1_line, pursuer_2_line, pursuer_3_line, *bars


# Create the animation
ani = FuncAnimation(fig, update, frames=len(ego.x),
                    init_func=init, blit=True, interval=40)

ax_3d.legend()
plt.tight_layout()
plt.show()
