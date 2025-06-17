import pathlib
import time
import torch
import numpy as np
import yaml
import os
import pickle as pkl
from typing import Dict, Any, List, Optional
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.transformers.wayformer.predictformer import PredictFormer
from jarvis.transformers.wayformer.dataset import LazyBaseDataset
from jarvis.envs.simple_agent import DataHandler, Pursuer, Evader
from jarvis.utils.vector import StateVector

from ray.rllib.core.rl_module import RLModule
from pytorch_lightning.callbacks import ModelCheckpoint
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)

from collections import deque

VEHICLE = 1
HEADING_IDX = 5
VELOCITY_IDX = 6

def recursive_to(obj, dev:str) -> Any:
    """
    Args:
        obj: The object to move to the device.
        dev: The device to move the object to, e.g., 'cuda' or 'cpu'.
    Returns:
        The object moved to the specified device.
    Recursively moves tensors, lists, tuples, and dictionaries to the specified device.
    """
    if torch.is_tensor(obj):
        return obj.to(dev)
    elif isinstance(obj, dict):
        return { k: recursive_to(v, dev) for k,v in obj.items() }
    elif isinstance(obj, (list, tuple)):
        # preserve type
        t = [recursive_to(v, dev) for v in obj]
        return type(obj)(t)
    else:
        return obj


def transform_with_current_heading(
    pred_traj: np.array, 
    current_heading: float, 
    current_position: np.array, 
    heading_index: int):
    """
    Transforms a predicted trajectory from the local frame back to the global frame 
    using the current ground truth heading and position.

    Args:
        pred_traj (np.ndarray): Predicted trajectory of shape [num_modes, T, num_attrs].
                                It is assumed that the first two attributes (0:2) are x,y offsets,
                                and the attribute at heading_index is the heading (in radians).
        current_heading (float): The current ground truth heading (in radians).
        current_position (np.ndarray): The current ground truth position (e.g. [x, y]).
        heading_index (int): The index of the heading feature in pred_traj.
        
    Returns:
        np.ndarray: The transformed (global) trajectory with the same shape as pred_traj.
    """
    # Compute cosine and sine of the current heading.
    c = np.cos(current_heading)
    s = np.sin(current_heading)
    
    # Option 1: Standard rotation matrix for +current_heading.
    R1 = np.array([[c, -s],
                [s,  c]])
    # Option 2: Alternative rotation matrix to correct flipping.
    R2 = np.array([[c,  s],
                [-s, c]])
    
    # Choose the matrix that gives correct orientation.
    # If your trajectories appear flipped, try using R2.
    R = R1

    # Extract local x,y coordinates (assumed to be in the first two columns).
    # pred_traj shape: [num_modes, T, num_attrs]
    local_xy = pred_traj[:, :, 0:2]
    
    # Rotate the local x,y coordinates by +current_heading.
    # Using np.einsum to multiply R with each [x, y] pair.
    rotated_xy = np.einsum('ij,mti->mtj', R, local_xy)
    
    # Translate by adding the current global position.
    global_xy = rotated_xy + current_position  # current_position should be shape [2]
    
    # Create a copy to avoid modifying the input.
    global_traj = pred_traj.copy()
    global_traj[:, :, 0:2] = global_xy
    
    # Adjust the heading feature: add the current heading back.
    global_traj[:, :, heading_index] += current_heading
    
    # Wrap the heading into the interval [-pi, pi]
    #global_traj[:, :, heading_index] = (global_traj[:, :, heading_index] + np.pi) % (2 * np.pi) - np.pi
        
    return global_traj

class DataHistory:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.data = deque(maxlen=max_length)

class RLSimHelper():
    """
    This class is used to help load the RL simulation environment
    used to train the RL model.
    Users must specify the simulation configuration environment
    """
    def __init__(self, checkpoint_path: str,
                 env_config:str, num_episodes: int = 1,
          use_pronav: bool = True, save: bool = False,
          index_save: int = 0, folder_dir: str = 'rl_pickle',
          env_type:str="pursuer_evader",
          use_predictformer:bool=False,
          predictformer_config:str = None) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.env_config: str = env_config
        self.num_episodes: int = num_episodes
        self.use_pronav: bool = use_pronav
        self.save: bool = save
        self.index_save: int = index_save
        self.folder_dir: str = folder_dir
        self.env_type: str = env_type
        self.use_predictformer: bool = use_predictformer
        self.predictformer_config: str = predictformer_config
        self.past_len: int = None
        self.future_len: int = None
        self.env: PursuerEvaderEnv = None
        self.predictformer_model: PredictFormer = None
        self.start_env()
        
        if self.use_predictformer:
            self.predictformer_model: PredictFormer = self.load_predictformer()
            # For storage we will preallocate a matrix of size [num_vehicles, past_len + future_len, 7]
            # where 7 is the number of attributes for each vehicle
            self.num_vehicles: int = 3  # 1 ego + 2 pursuers
            self.num_attributes: int = 7  # x, y, z, roll, pitch, yaw, speed
            self.time_history: List[float] = np.arange(
                0, self.past_len + self.future_len) * 0.1
            self.segment_info = np.zeros(
                (self.num_vehicles, 
                 self.past_len + self.future_len, 
                 self.num_attributes), dtype=np.float32)
    
    def start_env(self) -> None:
        """
        """
        if self.env_type == "pursuer_evader":
            self.env: PursuerEvaderEnv = self.create_multi_agent_env(
                env_config=self.env_config)

    def load_predictformer(self) -> PredictFormer:
        """
        To do - Load the PredictFormer model from the checkpoint
        and return the model in evaluation mode.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.predictformer_config is None:
            raise ValueError("PredictFormer configuration file is not provided.")
        
        model_config = self.predictformer_config
        with open(model_config, 'r') as f:
            model_config = yaml.safe_load(f)
            self.predictformer_config = model_config

        start_idx: int = model_config['past_len']
        self.past_len: int = start_idx
        self.future_len: int = model_config['future_len']
        #name = "high_speed_predictformer"
        name = "predictformer_21"
        # Check if there's an existing checkpoint to resume from
        checkpoint_dir = name+"_checkpoint/"

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

        if latest_checkpoint is None:
            raise FileNotFoundError(
                "No checkpoint found in the specified directory. "
                "Please ensure the checkpoint path is correct.")
        # set the model to evaluation mode
        model:PredictFormer = PredictFormer.load_from_checkpoint(
            latest_checkpoint, config=model_config)
        model.to(device)
        model.eval()
        
        return model
        
    def create_multi_agent_env(self,
        env_config: Dict[str, Any]) -> PursuerEvaderEnv:

        return PursuerEvaderEnv(
            config=env_config)


    def update_predictformer_observations(self, obs: Dict[str, Any]) -> None:
        """
        This function is used to update the observations for the PredictFormer model.
        It will update the overall_ego_position, overall_controls, and overall_pursuer_positions
        with the new observations.
        """
        # Assuming obs is a dictionary with keys 'ego' and 'pursuers'
        # where each key contains a list of past observations
        ego: np.array = np.array(obs['ego'],dtype=np.float32)
        pursuers: List[np.array] = obs['pursuers']
        
    def transform_trajs_to_center_coords(self, obj_trajs,
                                         center_xyz, center_heading,
                                         heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """

        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        # TODO: ADD ROTATION correctly based on the heading
        obj_trajs = np.tile(
            obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]
                  ] -= center_xyz[:, None, None, :]
        
        # For the x-y positions (first 2 coordinates), apply rotation using per-timestep heading.
        # points_xy shape: (B, num_objects, T, 2)
        # points_xy = obj_trajs[:, :, :, 0:2]
        # Rotate points using -center_heading (to align the center with zero heading).
        # rotated_xy = rotate_points_along_z(points_xy, -center_heading)
        # obj_trajs[:, :, :, 0:2] = rotated_xy
        # plot the trajectories

        
        # obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
        #     points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        #     angle=-center_heading
        # ).reshape(num_center_objects, num_objects, num_timestamps, 2)
        
        # # Assuming `heading_index` is the index of the heading feature.
        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, :]
        
        return obj_trajs

    def inverse_transform_trajs_from_center_coords(
        self, obj_trajs_center, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Inverse transforms trajectories from center (ego-centric) coordinates back to global coordinates.
        This reverses the transformation applied in transform_trajs_to_center_coords.

        Args:
            obj_trajs_center (np.ndarray): Trajectories in center coordinates with shape 
                (B, num_objects, T, num_attrs), where B = number of centers (agents).
            center_xyz (np.ndarray): Global center positions with shape (B, D), where D is 2 or 3.
            center_heading (np.ndarray): Center heading angles.
                Depending on your transform, this can be:
                - shape (B,) if the same heading is applied for all timesteps, or
                - shape (B, T) if per-timestep headings were used.
            heading_index (int): The index of the heading feature in obj_trajs_center.
            rot_vel_index (list, optional): List of indices for vector attributes (e.g. velocity) to rotate.

        Returns:
            np.ndarray: Global trajectories with shape (B, num_objects, T, num_attrs).
        """
        B, num_objects, T, num_attrs = obj_trajs_center.shape
        D = center_xyz.shape[1]  # Typically 2 (for x,y)

        # --- Inverse Rotation for x,y positions ---
        # Extract the x,y coordinates.
        points_xy = obj_trajs_center[:, :, :, 0:2]  # shape: (B, num_objects, T, 2)
        # If center_heading is given per agent (shape (B,)), expand it to per-timestep:
        if center_heading.ndim == 1:
            # Create a (B, T) array where each row is the same heading.
            center_heading = np.tile(center_heading[:, None], (1, T))
        
        # Build inverse rotation matrices for each center and timestep.
        # Since the original transform rotated by -center_heading, we now rotate by +center_heading.
        cos_vals = np.cos(center_heading)  # shape: (B, T)
        sin_vals = np.sin(center_heading)  # shape: (B, T)
        R_inv = np.empty((B, T, 2, 2))
        R_inv[:, :, 0, 0] = cos_vals
        R_inv[:, :, 0, 1] = -sin_vals
        R_inv[:, :, 1, 0] = sin_vals
        R_inv[:, :, 1, 1] = cos_vals

        # Apply the inverse rotation using np.einsum:
        rotated_xy = np.einsum('b t i j, b n t j -> b n t i', R_inv, points_xy)

        # --- Inverse Translation for x,y positions ---
        # Add back the center position.
        # Expand center_xyz from (B, D) to (B, 1, 1, D) for broadcasting.
        global_xy = rotated_xy #+ center_xyz[:, None, None, 0:2]

        # Prepare output by copying the input trajectories.
        global_trajs = obj_trajs_center.copy()
        global_trajs[:, :, :, 0:2] = global_xy

        # --- Inverse Heading Adjustment ---
        # The original transform subtracted center_heading from the heading feature.
        # Here, we add it back.
        # If center_heading is (B, T), we need to match dimensions:
        global_trajs[:, :, :, heading_index] += center_heading[:, None, :]

        # Wrap the heading to the interval [-pi, pi]
        # global_trajs[:, :, :, heading_index] = (global_trajs[:, :, :, heading_index] + np.pi) % (2 * np.pi) - np.pi
        # --- Optional: Inverse Rotation for Other Vector Attributes ---
        if rot_vel_index is not None:
            vel = global_trajs[:, :, :, rot_vel_index]  # shape: (B, num_objects, T, len(rot_vel_index))
            rotated_vel = np.einsum('b t i j, b n t j -> b n t i', R_inv, vel)
            global_trajs[:, :, :, rot_vel_index] = rotated_vel

        return global_trajs


    def get_agent_data(
            self,
            center_objects,
            obj_trajs_past,
            obj_trajs_future,
            track_index_to_predict,
            sdc_track_index,
            timestamps,
            obj_types):
        """
        Centers the location of all the agents 
        """
        center_objects = obj_trajs_past
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, num_attributes = obj_trajs_past.shape
        
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )
        obj_types = obj_types[0]
        object_onehot_mask = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(
            num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, num_timestamps))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps[:num_timestamps]

        object_heading_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(
            obj_trajs[:, :, :, HEADING_IDX])
        object_heading_embedding[:, :, :, 1] = np.cos(
            obj_trajs[:, :, :, HEADING_IDX])

        vel = obj_trajs[:, :, :, VELOCITY_IDX]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        # add another dimension to acce
        acce = np.expand_dims(acce, axis=-1)
        acce[:, :, 0, :] = acce[:, :, 1, :]
        expanded_velocity = np.expand_dims(
            obj_trajs[:, :, :, VELOCITY_IDX], axis=-1)

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:VELOCITY_IDX],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            expanded_velocity,
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0
    
        obj_trajs_future = obj_trajs_future.astype(np.float32)
        copy_obj = obj_trajs_future.copy()
        center_objects = obj_trajs_future
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )
        obj_trajs_future_test = obj_trajs_future[0,:,:,:]
        # fig, ax = plt.subplots()
        # # plot the 2d trajectories
        # for i in range(obj_trajs_future.shape[0]):
        #     ax.plot(copy_obj[i,:,0], copy_obj[i,:,1], label=f"Agent {i}")
        #     ax.plot(obj_trajs_future[i,i,:,0], obj_trajs_future_test[i,i,:,1], label=f"Agent {i} future")
        # ax.legend()
        # plt.show()

        # obj_trajs_future_state = obj_trajs_future[:, :, :, [
        #     0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_state = obj_trajs_future[:, :, :, [
            0, 1, 2, 3, VELOCITY_IDX]]  # (x, y, z, v)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs,
                                                 track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs,
                                                     track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(
            obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros(
            (num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:,
                                                               :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros(
            (num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.predictformer_config['max_num_agents']
        object_dist_to_center = np.linalg.norm(
            obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center,
                               axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(
            obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(
            obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(
            obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(
            obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(
            len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((
            0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(
            obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((
            0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def process(self, sim_data: Dict[str,Any],
                add_noise:bool=False) -> Dict[str,Any]:
        """
        Process the data in internal format and return the processed data.
        """
        # Process the data here.
        idx_to_track: int = sim_data['idx_to_track']

        timestamp = sim_data['timestamp']
        obj_trajs_full: np.array = sim_data['segment_data']
        obj_trajs_full[:, :, HEADING_IDX] = np.deg2rad(
            obj_trajs_full[:, :, HEADING_IDX])
        obj_types: List[int] = sim_data['object_type']
        obj_trajs_past: np.array = obj_trajs_full[:, :self.past_len, :]
        obj_trajs_future: np.array = obj_trajs_full[:,
                                                    self.past_len:, :]
        
        track_idx_to_predict = [i for i in range(len(obj_trajs_full))]
        center_objects = obj_trajs_full
        original_pos_past: np.array = obj_trajs_past.copy()
        
        # ========================== NOISE INJECTION ==========================

        ## 1. Measurement Noise (Sensor Errors)
        # Gaussian position noise (simulating GPS or LIDAR errors)
        if add_noise:
            position_noise:float = 0.2
            obj_trajs_past[:, :, 0:2] += np.random.normal(0, position_noise, obj_trajs_past[:, :, 0:2].shape)  # (Mean 0, Std 0.1m)
        
            # # Multiplicative noise (simulating sensor drift)
            obj_trajs_past[:, :, 0:2] *= np.random.normal(1, 
                                                        0.01, 
                                                        obj_trajs_past[:, :, 0:2].shape)  # 2% variation

            # Heading noise (simulating IMU/Gyro errors)
            # obj_trajs_past[:, :, HEADING_IDX] += np.random.normal(0, np.deg2rad(1), obj_trajs_past[:, :, HEADING_IDX].shape)  # 2-degree noise

            ## 2. Process Noise (Motion Model Uncertainty)
            # Random walk noise (simulating object drift over time)
            # drift = np.cumsum(np.random.normal(0, 0.05, obj_trajs_past[:, :, 0:2].shape), axis=1)  # Accumulate small movements
            # obj_trajs_past[:, :, 0:2] += drift

            # Velocity noise (simulating varying acceleration)
            # velocity_noise = np.random.normal(0, 0.2, obj_trajs_past[:, :, VELOCITY_IDX].shape)  # Velocity in (x, y)
            # obj_trajs_past[:, :, VELOCITY_IDX] += velocity_noise

        else:
            obj_trajs_past = obj_trajs_past.astype(np.float32)
            
        (obj_trajs_data, obj_trajs_mask, 
        obj_trajs_pos, obj_trajs_last_pos, 
        obj_trajs_future_state,
        obj_trajs_future_mask, center_gt_trajs,
        center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new) = self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_idx_to_predict,
            sdc_track_index=idx_to_track,
            timestamps=timestamp, obj_types=obj_types
        )
                    
        ret: Dict[str, Any] = {
            # 'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            # used to select center-features
            'track_index_to_predict': track_index_to_predict_new,
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            # 'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(obj_types),
            # 'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_idx_to_predict],
            'original_pos_past': original_pos_past,
        }
        
        return ret

        
    def process_segment(self, segment: np.array, timestamps: List[float], idx: int) -> Dict[str, Any]:
        """
        Processes a single segment.
        For instance, here we convert the heading angles from degrees to radians.
        You can extend this method to perform additional processing as required.
        """
        # Convert heading (at HEADING_IDX) from degrees to radians.
        # segment[:, :, HEADING_IDX] = np.deg2rad(segment[:, :, HEADING_IDX])
        assert segment.ndim == 3
        ego_idx:int = 0        
        tracks_to_predict: Dict[str, Any] = {
            'track_index': [],
            'object_type': []
        }
        num_pursuers, total_steps, _ = segment.shape

        # Create and return a dictionary for the processed segment.
        processed = {
            'object_type': [],
            # 'idx': [],
            'timestamp': timestamps,
            'idx_to_track': ego_idx,
            'segment_idx': idx,
            'segment_data': segment,
            # Add any additional keys for further processed outputs.
        }
        
        num_ego: int = 1
        total_agents:int = num_pursuers + num_ego
        for i in range(total_agents):
            tracks_to_predict['track_index'].append(i)
            tracks_to_predict['object_type'].append(VEHICLE)
            processed['object_type'].append(VEHICLE)            

        processed['tracks_to_predict'] = tracks_to_predict
                
        return processed
    
    def batch_data(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function (if using PyTorch DataLoader).
        
        Process will return past_len + future_len segments
        
        """
        processed_data: List[Dict[str, Any]] = []
        processed_data:List[Dict[str,Any]] = [self.process(sample) for sample in data_list]
        input_dict = {}
        for key in processed_data[0].keys():
            input_dict[key] = torch.from_numpy(np.stack([sample[key] for sample in processed_data]))
            
        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()
        
        batch_list = []
        for batch in data_list:
            batch_list += batch
            
        batch_size = len(batch_list)
        
        batch_dict = {
            'batch_size': batch_size,
            'input_dict': input_dict,
            'batch_sample_count': batch_size
        }
        
        return batch_dict

    def get_predictformer_batch(self, obs: Dict[str, Any], time_step:float,
                                current_counter:int) -> Dict[str, Any]:
        """
        This function is used to get the batch of observations
        for the PredictFormer model, if using the simple environment recall the 
        observations contains the following information:
        - x: The x position of the agent
        - y: The y position of the agent
        - z: The z position of the agent
        - roll: The roll of the agent
        - pitch: The pitch of the agent
        - yaw: The yaw of the agent
        - speed: The speed of the agent
        

        Overall ego position should be shaped (num, 7) where 7 is 
        The pursuers are shaped:
            - [num_pursuers, num_past_observations, 7]
        
        """
        # Assuming obs is a dictionary with keys 'ego' and 'pursuers'
        # where each key contains a list of past observations
        # Convert to tensors and move to device
        obs["time_step"] = time_step
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # batch = recursive_to(batch, device)
        # for predictformer to work we need to have the past n observations
        if self.past_len is None or self.future_len is None:
            raise ValueError("PredictFormer past_len and future_len are not set.")

        if current_counter <= self.past_len:
            ego: np.array = obs["ego"]
            pursuers: List[np.array] = obs["vehicles"]
            self.segment_info[0, current_counter, :] = ego
            for i, pursuer in enumerate(pursuers):
                self.segment_info[i+1, current_counter, :] = pursuer
        else:
            # if we have more than past_len observations, we need to pop the oldest observation
            # and update the segment_info
            self.segment_info[:, :-1, :] = self.segment_info[:, 1:, :]
            ego: np.array = obs["ego"]
            pursuers: List[np.array] = obs["vehicles"]
            
            # we index the segment info to past_len index since it is the most recent observation
            # everything forward on is zeros because we really don't have any future observations
            # in real time              
            self.segment_info[0, self.past_len, :] = ego
            for i, pursuer in enumerate(pursuers):
                self.segment_info[i+1, self.past_len, :] = pursuer
        
        processed_segment:Dict[str,Any] = self.process_segment(
            segment=self.segment_info,
            timestamps=self.time_history,
            idx=current_counter)
        
        return self.batch_data([processed_segment])

    def place_pursuers(self) -> None:
        """
        Place pursuers in the environment.
        """
        self.env.remove_all_agents()
        evader_x: float = 0.0#np.random.uniform(-1, 1)
        evader_y: float = 0.0#np.random.uniform(-1, 1)
        evader_z: float = 55 # np.random.uniform(55, 56)
        state_vector = StateVector(
            x=evader_x, y=evader_y, z=evader_z, yaw_rad=np.deg2rad(90), roll_rad=0,
            pitch_rad=0, speed=15)
        
        evader: Evader = Evader(
            agent_id="0",
            state_vector=state_vector,
            battle_space=self.env.battlespace,
            simple_model=PlaneKinematicModel(),
            is_controlled=True,
            radius_bubble=5,
        )
        
        rand_x: float = -55 #np.random.uniform(-55, -54)
        rand_y: float = 175#np.random.uniform(175, 176)
        rand_z: float = 55 #np.random.uniform(55, 56)
        state_vector = StateVector(
            x=rand_x, y=rand_y, z=rand_z, yaw_rad=np.deg2rad(270), roll_rad=0,
            pitch_rad=0, speed=20)
        
        pursuer: Pursuer = Pursuer(
            agent_id="1",
            state_vector=state_vector,
            battle_space=self.env.battlespace,
            simple_model=PlaneKinematicModel(),
            is_controlled=True,
            radius_bubble=5
        )

        rand_x: float = 50 #np.random.uniform(49, 50)
        rand_y: float = 175 #np.random.uniform(175, 176)
        rand_z: float = 55 #np.random.uniform(55, 56)
        state_vector = StateVector(
            x=rand_x, y=rand_y, z=rand_z, yaw_rad=np.deg2rad(270), roll_rad=0,
            pitch_rad=0, speed=20)

        pursuer_2: Pursuer = Pursuer(
            agent_id="2",
            state_vector=state_vector,
            battle_space=self.env.battlespace,
            simple_model=PlaneKinematicModel(),
            is_controlled=True,
            radius_bubble=5
        )

        self.env.insert_agent(evader)
        self.env.insert_agent(pursuer)
        self.env.insert_agent(pursuer_2)

        self.env.init_action_space()
        self.env.init_observation_space()
        
    def infer_pursuer_evader_env(self,
                                 head_on_placement:Optional[bool]=False,
                                 use_predictformer_for_rl:Optional[bool]=False,
                                 desired_time_idx:int = 1) -> None:
        """
        This function is used to infer the pursuer evader environment
        and run the simulation for the specified number of episodes.
        """
        checkpoint_path: str = self.checkpoint_path
        env:PursuerEvaderEnv = self.env
        
        # load the model from our checkpoints
        # Create only the neural network (RLModule) from our checkpoint.
        evader_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["evader_policy"]

        pursuer_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["pursuer_policy"]

        reward_list: List[float] = []
        terminated = {'__all__': False}
        observation, info = env.reset()
        
        if head_on_placement:
            self.place_pursuers()
        
        counter = 0
        
        time_step:float = 0.0
        current_counter:int = 0
        dt:float = 0.1  # time step in seconds
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        end_counter:int = 1200
        
        predicted_traj_list: List[np.array] = []
        observation_history: List[Dict[str, Any]] = []
        batch_history: List[Dict[str, Any]] = []
        output_history: List[Dict[str, Any]] = []
        original_pos_history: List[np.array] = []
        predictformer_count:int = 0 # used to keep track of trajectory stored
        
        while not terminated['__all__']:
            print(f"Time step: {env.current_step}, Counter: {current_counter}")
            
            if self.use_predictformer:
                # for this to work we need the past n observations to predict the future
                predictformer_obs:Dict[str,Any] = self.env.get_observation_from_predictformer()
                batch = self.get_predictformer_batch(
                    obs=predictformer_obs, time_step=time_step,
                    current_counter=current_counter)
                batch = recursive_to(batch, dev=device)
                output_cuda, _ = self.predictformer_model(batch)
                batch = recursive_to(batch, 'cpu')
                output = recursive_to(output_cuda, 'cpu')
                del output_cuda
                center_xyz = batch['input_dict']['center_objects_world'].detach().numpy()
                # convert to cpu
                # cpu_traj = (
                #     batch['input_dict']['center_gt_trajs']
                #     .detach()      # break the graph
                #     .cpu()         # copy to host (CPU) memory
                #     .numpy()       # now safe to convert
                # )
                # cpu_objs = (
                #     batch['input_dict']['center_objects_world']
                #         .detach()   # unhook from the graph
                #         .cpu()      # move tensor from GPU to CPU
                #         .numpy()    # now safe to convert to a NumPy array
                # )
                output['input_obj_trajs'] = (
                    batch['input_dict']['obj_trajs']
                        .detach()     # break the computation graph
                        .cpu()        # move tensor from GPU to CPU
                        .numpy()      # convert to NumPy
                        .squeeze()    # remove any singleton dimensions
                )
                
                original_pos_past = (
                    batch['input_dict']['center_objects_world']
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                )

                predicted_traj = output['predicted_trajectory'].detach().numpy()
                # center_xy = center_xyz.squeeze()[:, self.past_len, 0:2]
                # center_heading = center_xyz.squeeze()[:, self.past_len, 5]
                # predicted_headings = predicted_traj[:, :, 5]
                # predicted_ground_traj = self.inverse_transform_trajs_from_center_coords(
                #     obj_trajs_center=predicted_traj,
                #     center_xyz=center_xy,
                #     center_heading=center_heading,
                #     heading_index=5
                # )
                output['predicted_ground_traj'] = predicted_traj
                
                # if counter is a modulus of 3 save
                if current_counter % self.num_vehicles == 0:                  
                    predicted_traj_list.append(predicted_traj)
                    observation_history.append(
                        self.env.get_observation_from_predictformer())
                    predictformer_count += 1
                    print(f"PredictFormer count: {predictformer_count}")               
                    # batch_history.append(cpu_objs)
                    # output_history.append(output)
                    original_pos_history.append(original_pos_past)
        
            key_value = list(observation.keys())[0]
            if key_value == '1':
                obs = observation['1']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
                
            elif key_value == '0':
                obs = observation['0']             
                # If using transformer we need to pack the observations and batch it correctly
                # to feed into the transformer model
                if predictformer_count > self.past_len and use_predictformer_for_rl:
                    # replace the of the bad guys with the predictformer batch
                    all_obs:Dict[str,Any] = self.env.get_observation_from_predictformer()
                    all_pursuers:List[np.array] = all_obs['vehicles']
                    ego:np.array = all_obs['ego']
                    x_idx:int = 0
                    y_idx:int = 1
                    z_idx:int = 2
                    heading_idx:int = 5
                    vel_idx:int = 6
                    vx_idx:int = 7
                    vy_idx:int = 8
                    vz_idx:int = 9
                    # recall the dt is 0.1 seconds
                    desired_time_step_idx: int = 25
                    for i, pursuer in enumerate(all_pursuers):
                        pursuer_pos:np.array = pursuer[x_idx:z_idx]
                        pursuer_heading:float = pursuer[heading_idx]
                        pursuer_3d_pos:np.array = pursuer[x_idx:z_idx+1]
                        idx_pursuer = i + 1  # +1 because the first agent is the evader
                        pursuer_predicted_traj:np.array = predicted_traj[idx_pursuer]
                        # transform the predicted trajectory with the current heading
                        transformed_traj:np.array = transform_with_current_heading(
                            pred_traj=pursuer_predicted_traj,
                            current_heading=pursuer_heading,
                            current_position=pursuer_pos,
                            heading_index=heading_idx
                        )
                        closest_idx:int = 1
                        closest_distance:float = float('inf')
                        
                        # for simplicity sake we want to choose the location that is closest to our evader from our GMMs
                        for j in range(transformed_traj.shape[0]):
                            x = transformed_traj[j, desired_time_step_idx, x_idx]
                            y = transformed_traj[j, desired_time_step_idx, y_idx]
                            z = transformed_traj[j, desired_time_step_idx, z_idx]
                            z = z + pursuer_3d_pos[2]  # add the z position of the pursuer
                            # print("predicted position: ", x, y, z)
                            # print("pursuer position: ", pursuer_3d_pos)
                            distance = np.linalg.norm(
                                np.array([x, y, z]) - ego[x_idx:z_idx+1])
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_idx = j
                        
                        #TODO: There's a bug here not predicting the actual distance correctly 
                        # the predicted distance should be closer than the actual distance
                        # print("distance: ", distance)
                        # actual_distance:float = np.linalg.norm(
                        #     np.array([ego[x_idx], ego[y_idx], ego[z_idx]]) -
                        #     pursuer_3d_pos)
                        # print("actual distance: ", actual_distance)
                        ## For this highest predicted threat from our gmm we want to update our observations of the
                        # pursuers with this informatio
                        highest_threat:np.array = transformed_traj[closest_idx]
                        rel_x:float = ego[x_idx] - highest_threat[desired_time_step_idx, x_idx]

                        rel_y:float = ego[y_idx] - highest_threat[desired_time_step_idx, y_idx]
                        rel_z:float = ego[z_idx] - highest_threat[desired_time_step_idx, z_idx] 
                        rel_heading:float = ego[heading_idx] - np.deg2rad(highest_threat[desired_time_step_idx, heading_idx])
                        # wrap the heading to [-pi, pi]
                        rel_heading:float = (rel_heading + np.pi) % (2 * np.pi) - np.pi
                        rel_vel:float = ego[vel_idx] - highest_threat[desired_time_step_idx, vel_idx]
                       
                        # now we have the closest index, we can replace the pursuer observationers
                        offset_counter = i*5  # since we have 5 observations per pursuer 
                        from copy import deepcopy
                        old_observations = deepcopy(obs['observations'])
                        print("old x position: ", old_observations[vz_idx+offset_counter+1])

                        print("old y position: ", old_observations[vz_idx+offset_counter + 2])
                        obs['observations'][vz_idx+offset_counter + 1] = rel_x
                        obs['observations'][vz_idx+offset_counter + 2] = rel_y
                                                # print("new x position: ", obs['observations'][vz_idx+offset_counter+1])

                        print("new y position: ", obs['observations'][vz_idx+offset_counter + 2])
                        # obs['observations'][vz_idx+offset_counter + 2] = rel_z
                        # obs['observations'][vz_idx+offset_counter + 3] = rel_heading
                        # obs['observations'][vz_idx+offset_counter + 4] = rel_vel
                    
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = evader_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
                
            elif key_value == '2':
                obs = observation['2']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]

            # For my action space I have a multidscrete environment
            # Since my action logits are a [1 x total_actions] tensor
            # I need to get the argmax of the tensor
            
            # print("time: ", end_time - start_time)
            action_logits = action_logits.detach().numpy().squeeze()
            unwrapped_action: Dict[str, np.array] = env.unwrap_action_mask(
                action_logits)

            discrete_actions = []
            for k, v in unwrapped_action.items():
                v = torch.from_numpy(v)
                best_action = torch.argmax(v).numpy()
                discrete_actions.append(best_action)
            
            # action = torch.argmax(action_logits).numpy()
            action_dict = {}
            action_dict[key_value] = {'action': discrete_actions}
            # print("action dict: ", action_dict)

            observation, reward, terminated, truncated, info = env.step(
                action_dict=action_dict)

            reward_list.append(reward)

            # check if done
            if (terminated['__all__'] == True):
                # print("reward: ", reward)
                break
            
            time_step += dt
            # current_counter += 1
            current_counter = env.current_step
            
            if current_counter >= end_counter:
                print("Reached maximum number of steps, ending episode.")
                break

        # store the predicted trajectory
        history: Dict[str, Any] = {
            'predicted_traj': predicted_traj_list,
            'observation_history': observation_history,
            # 'batch_history': batch_history,
            # 'output_history': output_history,
            'reward_list': reward_list,
            'time_step': time_step,
            'current_counter': current_counter,
            'original_pos_history': original_pos_history,
        }
        
        # store history as pickle file
        with open('postprocess_predictformer/pursuer_evader_history.pkl', 'wb') as f:
            pkl.dump(history, f)
        print("History saved to pursuer_evader_history.pkl")