import numpy as np
import unittest
import yaml
import torch
import matplotlib.pyplot as plt
import time 
from jarvis.transformers.wayformer.dataset import LazyBaseDataset as BaseDataset
from jarvis.transformers.wayformer.predictformer import PredictFormer
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os


plt.close('all')    

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
data_config = "config/high_speed_predictformer_config_small.yaml"
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)

dataset = BaseDataset(
    config=data_config,
    is_validation=False,
    num_samples=100)
print("Dataset Length", len(dataset))
# set seed numberEW
# seed = 42
# torch.manual_seed(seed)
dataloader: DataLoader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

model_config: str = "config/high_speed_predictformer_config_small.yaml"
with open(model_config, 'r') as f:
    model_config = yaml.safe_load(f)

start_idx: int = data_config['past_len']
name = "high_speed_predictformer_small_" + str(start_idx) 
# Check if there's an existing checkpoint to resume from
checkpoint_dir = name+"_checkpoint/"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="uavt-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f)
         for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(
            f"Resuming training from checkpoint: {latest_checkpoint}")


# set the model to evaluation mode
model = PredictFormer.load_from_checkpoint(
    latest_checkpoint, config=model_config)
model.to(device)
model.eval()

# test a batch
# During testing, move batch tensors to the same device

batch_history = []
output_history = []
center_gt_trajs = []
center_objects_world = []
infer_time = []
for i, batch in enumerate(dataloader):
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    start_time = time.time()
    output, loss = model(batch)
    end_time = time.time()
    print(f"Time taken for inference: {end_time - start_time}")
    # if i == 2:
    #     break
    center_gt_trajs.append(batch['input_dict']['center_gt_trajs'].detach().numpy())
    center_objects_world.append(batch['input_dict']['center_objects_world'].detach().numpy())
    predicted_traj = output['predicted_trajectory'].detach().numpy()
    center_xyz = batch['input_dict']['center_objects_world'].detach().numpy()
    center_xy = center_xyz.squeeze()[:, start_idx, 0:2]
    center_heading = center_xyz.squeeze()[:, start_idx, 5]
    predicted_headings = predicted_traj[:, :, 5]
    predicted_ground_traj = dataset.inverse_transform_trajs_from_center_coords(
        obj_trajs_center=predicted_traj,
        center_xyz=center_xy,
        center_heading=center_heading,
        heading_index=5
    )

    output['input_obj_trajs'] = batch['input_dict']['obj_trajs'].detach().numpy().squeeze()
    output['predicted_ground_traj'] = predicted_traj
    new_output = {}
    
    # convert the output to numpy
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            new_output[key] = value.detach().numpy()
        else:
            new_output[key] = value
    
    output_history.append(new_output)
    infer_time.append(end_time - start_time)
    if i == 10:
        break

# #Pickkle the output and batch
import pickle as pkl
info = {"output": output_history,
        "infer_time": infer_time,
        "center_gt_trajs": center_gt_trajs,
        "center_objects_world": center_objects_world}
folder_dir = "postprocess_predictformer"
if not os.path.exists(folder_dir):
    os.makedirs(folder_dir)
pkl.dump(info, open(os.path.join(folder_dir, "small_model.pkl"), "wb"))


# %%
# Predicited probability is an [num_agents, num_modes] num_modes is the gaussian mixture model
predicted_probability: np.array = output['predicted_probability'].detach(
).numpy()
# predicted trajectory is [num_agents, num_modes, num_timesteps, num_attributes]
predicted_trajectory: np.array = output['predicted_trajectory'].detach(
).numpy()
predicted_traj = output['predicted_ground_traj']

num_modes: int = predicted_probability.shape[1]
ground_truth_trajectory: np.array = batch['input_dict']['center_gt_trajs'].squeeze(
).detach().numpy()

ground_truth_world = batch['input_dict']['center_objects_world'].squeeze(
).detach().numpy()
original_pos_past = batch['input_dict']['center_objects_world'].squeeze().detach().numpy()
mask = batch['input_dict']['center_gt_trajs_mask'].unsqueeze(-1)


num_agents: int = predicted_probability.shape[0]
# Let's plot each agent trajectory in a seperate plot and show the gaussian mixture model trajectory of the agent
for i in range(num_agents):
    fig, ax = plt.subplots(1, 1)
    # this becomes [num_modes, num_timesteps, num_attributes]
    agent_traj: np.array = predicted_trajectory[i]
    agent_probability: np.array = predicted_probability[i]
    for j in range(num_modes):
        highest_probabilty_index = np.argmax(agent_probability)
        x = agent_traj[j, :, 0]
        y = agent_traj[j, :, 1]
        ax.scatter(
            x, y, label=f"Mode {j} for agent {i} ")
    ax.plot(ground_truth_trajectory[i, :, 0],
            ground_truth_trajectory[i, :, 1], label="Ground Truth")
    # set the title
    ax.set_title(
        f"Agent {i} Trajectory Highest Probability Mode {highest_probabilty_index}")
    ax.legend()

fig, ax = plt.subplots(1, 1)
heading_idx:int = 5
# transpose this to ground truth trajectory [num_agents, num_timesteps, num_attributes]
for i in range(num_agents):
    x = original_pos_past[i, :, 0]
    y = original_pos_past[i, :, 1]
    x_start = x[start_idx]
    y_start = y[start_idx]
    ax.plot(x, y, label="Ground Truth " + str(i))    
    agent_traj = predicted_traj[i]
    current_heading = original_pos_past[i, start_idx, heading_idx]
    print("current heading", np.rad2deg(current_heading))
    current_position = original_pos_past[i, start_idx, :2]
    transformed_traj = dataset.transform_with_current_heading(
        pred_traj=agent_traj,
        current_heading=current_heading,
        current_position=current_position,
        heading_index=heading_idx
    )
    for j in range(num_modes):
        highest_probabilty_index = np.argmax(predicted_probability[i])
        x = x_start + agent_traj[j, :, 0]
        y = y_start + agent_traj[j, :, 1]
        # x = transformed_traj[j, :, 0]
        # y = transformed_traj[j, :, 1]
        ax.scatter(
            x, y, label=f"Mode {j} for agent {i} ")

    ax.scatter(x_start, y_start, label="Start " + str(i))
    ax.legend()

# Now plot this for 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(num_agents):
    x = original_pos_past[i, :, 0]
    y = original_pos_past[i, :, 1]
    z = original_pos_past[i, :, 2]
    x_start = x[start_idx]
    y_start = y[start_idx]
    z_start = z[start_idx]
    
    agent_traj = predicted_traj[i]
    current_heading = original_pos_past[i, start_idx, heading_idx]
    current_position = original_pos_past[i, start_idx, :2]
    transformed_traj = dataset.transform_with_current_heading(
        pred_traj=agent_traj,
        current_heading=current_heading,
        current_position=current_position,
        heading_index=heading_idx
    )
    ax.plot(x, y, z, label="Ground Truth " + str(i))
    agent_traj = predicted_trajectory[i]
    for j in range(num_modes):
        highest_probabilty_index = np.argmax(predicted_probability[i])
        x = x_start + agent_traj[j, :, 0]
        y = y_start + agent_traj[j, :, 1]
        z = z_start + agent_traj[j, :, 2]
        # x = transformed_traj[j, :, 0]
        # y = transformed_traj[j, :, 1]
        
        
        ax.scatter(
            x, y, z, label=f"Mode {j} for agent {i} ")

    ax.scatter(x_start, y_start, z_start, label="Start " + str(i))
    ax.legend()

#%%
plt.show()
