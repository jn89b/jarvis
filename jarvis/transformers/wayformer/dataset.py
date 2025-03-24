from typing import Dict, Any, List,Tuple
from omegaconf import OmegaConf
import os
import pickle


import shutil
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pickle as pkl
import glob
import torch
import json
# from metadrive.scenario.scenario_description import MetaDriveType
# from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
# TODO: Reclass as Enum
VEHICLE = 1
HEADING_IDX = 5
VELOCITY_IDX = 6


def rotate_points_along_z(points, angles):
    """
    Rotate points along the Z-axis using a 2D rotation matrix for each center object and timestep.
    
    Args:
        points (np.ndarray): Shape (B, N, T, 2), where
            B = number of center objects,
            N = number of objects,
            T = number of timesteps.
        angles (np.ndarray): Shape (B, T) containing rotation angles in radians 
                             (e.g. -center_heading per center per timestep).
                             
    Returns:
        np.ndarray: Rotated points with shape (B, N, T, 2).
    """
    B, N, T, _ = points.shape
    # Build a rotation matrix for each center and timestep:
    # R has shape (B, T, 2, 2)
    cos_vals = np.cos(angles)  # shape: (B, T)
    sin_vals = np.sin(angles)  # shape: (B, T)
    R = np.empty((B, T, 2, 2))
    R[:, :, 0, 0] = cos_vals
    R[:, :, 0, 1] = -sin_vals
    R[:, :, 1, 0] = sin_vals
    R[:, :, 1, 1] = cos_vals

    # Use np.einsum to perform batch multiplication:
    # For each center (B) and each timestep (T), rotate each object's 2D point.
    rotated_points = np.einsum('b t i j, b n t j -> b n t i', R, points)
    return rotated_points
    

def generate_mask(current_index: int, total_length: int, interval: int) -> np.array:
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)


class BaseDataset(Dataset):

    def __init__(self, config=None,
                 is_test: bool = False,
                 is_validation=False,
                 num_samples=None):

        self.is_validation: bool = is_validation
        if is_test:
            print("Loading test data")
            self.data_path = config['test_data_path']
        elif is_validation:
            self.data_path = config['val_data_path']
        else:             
            self.data_path: str = config['train_data_path']

        self.config = config
        self.starting_frame: int = 0
        self.data_loaded_memory = []
        self.data_chunk_size: int = 8
        self.num_samples: int = num_samples
        # self.load_data()
        self.past_len: int = self.config['past_len']
        self.preprocess_data: List[Dict[str, Any]] = self.segment_data()
        self.process_data: List[Dict[str, Any]
                                ] = self.process(self.preprocess_data)
        print("You have successfully loaded the data")

    def __len__(self):
        # TODO: update this process data will be not a list in the future
        return len(self.process_data)

    def split_data(self, traj_data: np.array,
                   future_len: int, step_size: int,
                   time_interval: List[float]) -> Dict[str, Any]:
        """
        Splits a trajectory into overlapping segments.

        Args:
            traj: np.ndarray of shape (num_vehicles, T, features), where T is the number of timesteps.
            past_len: Number of past steps to consider.
            future_len: Number of future steps for prediction.
            step_size: Sliding window step size.

        Returns:
            Dictionary with keys 'idx', 'segmented_idx', 'data'.
            - 'idx': List of segment indices.
            - 'segmented_idx': List of starting indices of segments.
            - 'data': np.ndarray of shape (num_segments, 
                num_vehicles, past_len + future_len, features).
        """
        # check if dimension is correct
        assert traj_data.ndim == 3
        num_pursuers, total_steps, _ = traj_data.shape
        ego_idx: int = 0
        total_len = self.past_len + future_len
        segments = []
        # for start_idx in range(self.past_len, total_steps - total_len + 1, step_size):
        #     # if the start index is 0, then the segment
        #     segment = traj_data[:, start_idx:start_idx + total_steps, :]
        #     segments.append(segment)
        sim_data: Dict[str, Any] = {
            'object_type': [],
            'idx': [],
            'timestamp': [],
            'idx_to_track': ego_idx,
            'segmented_idx': [],
            'segment_data': []
        }
        tracks_to_predict: Dict[str, Any] = {
            'track_index': [],
            'object_type': []
        }

        idx_counter = 0
        # segment_data will consist of a list of np.arrays
        # where each np.array is of size [num_objects, total_len, num_attributes]
        for start_idx in range(total_len, total_steps - total_len, step_size):

            segment = traj_data[:, start_idx - total_len:start_idx, :]
            segments.append(segment)

            sim_data['idx'].append(idx_counter)
            sim_data['segmented_idx'].append(
                start_idx - total_len)
            idx_counter += 1
            sim_data['timestamp'].append(
                time_interval[start_idx: start_idx + total_len])

        num_ego: int = 1
        total_agents = num_pursuers + num_ego

        # for now just put them all as 1
        for i in range(total_agents):
            tracks_to_predict['track_index'].append(i)
            tracks_to_predict['object_type'].append(VEHICLE)
            sim_data['object_type'].append(VEHICLE)

        sim_data['segment_data'] = np.array(segments)
        sim_data['tracks_to_predict'] = tracks_to_predict

        return sim_data

    def segment_data(self) -> None:
        """
        Look into a directory, create that massive array and segment it into chunks
        Returns:
            preprocess_data: List[Dict[str, Any]]: a list of dictionaries containing the segmented data 
        """
        print("file path: ", self.data_path)
        if self.is_validation:
            print("Loading validation data...")
        else:
            print("Loading training data...")

        # Get all JSON files
        json_files: List[str] = glob.glob(
            os.path.join(self.data_path, "*.json")
        )

        if self.num_samples is not None:
            json_files = json_files[:self.num_samples]

        # TODO: Parallelize this
        # Okay for each JSON file what we are going to do is:
        # 1. Load the JSON file
        # We're going to loop through each timestep of the list which is a dictionary
        # # Append the information from each of the keys to a massive array
        # # Once we have the massive array, we're going to segment it into chunks
        # # Save the chunks into a temporary directory as a pkl file and use it to load the data
        # This massive array will be size: [num_objects, length_time, num_attributes]
        preprocess_data: List[Dict[str, Any]] = []
        for j_file in json_files:
            with open(j_file, 'r') as f:
                sim_data: List[Dict[str, Any]] = json.load(f)
            # close the file
            overall_ego_position: List[List[float]] = []
            overall_controls: List[List[float]] = []
            overall_timestamps: List[float] = []
            num_other_vehicles: int = len(sim_data[0]['vehicles'])

            # The length of this list is number of pursuers
            # where each pursuer is size [length_time, num_attributes]
            overall_pursuer_positions: List[np.array] = []

            for i in range(num_other_vehicles):
                overall_pursuer_positions.append([])

            # Process the individual simulation data
            for i, current_info in enumerate(sim_data):
                ego_position: List[float] = current_info['ego']
                controls: List[float] = current_info['controls']
                overall_ego_position.append(ego_position)
                overall_controls.append(controls)
                overall_timestamps.append(current_info['time_step'])
                for j, pursuer_info in enumerate(current_info['vehicles']):
                    overall_pursuer_positions[j].append(pursuer_info)

            # this is where I want create the fat arrays
            # From appending I'm going to get a list for the following:
            # - overall_ego_position
            # - overall_controls
            # - overall_pursuer_positions a list of lists containing the positions of the pursuers
            # convert them all into a numpy array
            overall_ego_position: np.array = np.array(overall_ego_position)
            overall_controls: np.array = np.array(overall_controls)
            for i, veh in enumerate(overall_pursuer_positions):
                veh: np.array = np.array(veh)
                overall_pursuer_positions[i] = veh

            # Right now the overall_pursuer positions is a list
            # lets stack into an array of size [num_pursuers, length_time, num_attributes]
            overall_pursuer_positions = np.stack(overall_pursuer_positions)

            # overall trajectory becomes a numpy array of size [num_objects, length_time, num_attributes]
            overall_traj: np.array = np.vstack(
                [overall_ego_position[np.newaxis, ...],
                 overall_pursuer_positions])
            sim_traj: Dict[str, Any] = self.split_data(
                overall_traj, self.config['future_len'], 1, overall_timestamps)

            preprocess_data.append(sim_traj)

        return preprocess_data

    def load_data(self):
        """
        TODO: Need to multi-process this function
        """
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')

        for cnt, data_path in enumerate(self.data_path):
            print(f'Loading data from {data_path}')
            self.starting_frame = self.config['starting_frame'][cnt]

        self.process_data_chunk(0)

    def process_data_chunk(self, worker_index: int):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        output_buffer = []
        save_cnt = 0
        for cnt, file_name in enumerate(data_list):
            if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                print(f'{cnt}/{len(data_list)} data processed', flush=True)

            # TODO: NEED TO change this
            scenario = read_scenario(data_path, mapping, file_name)

            try:
                # pickle the scenario
                with open('scenario.pkl', 'wb') as f:
                    pickle.dump(scenario, f)

                output = self.preprocess(scenario)

                with open('preprocess.pkl', 'wb') as f:
                    pickle.dump(output, f)
                output = self.process(output)

                with open('process.pkl', 'wb') as f:
                    pickle.dump(output, f)

                output = self.postprocess(output)
                with open('postprocess.pkl', 'wb') as f:
                    pickle.dump(output, f)

            except Exception as e:
                print('Error: {} in {}'.format(e, file_name))
                output = None

            if output is None:
                continue

            output_buffer += output

            while len(output_buffer) >= self.data_chunk_size:
                save_path = os.path.join(
                    self.cache_path, f'{worker_index}_{save_cnt}.pkl')
                to_save = output_buffer[:self.data_chunk_size]
                output_buffer = output_buffer[self.data_chunk_size:]
                with open(save_path, 'wb') as f:
                    pickle.dump(to_save, f)
                save_cnt += 1
                file_info = {}
                kalman_difficulty = np.stack(
                    [x['kalman_difficulty'] for x in to_save])
                file_info['kalman_difficulty'] = kalman_difficulty
                file_info['sample_num'] = len(to_save)
                file_list[save_path] = file_info

        save_path = os.path.join(
            self.cache_path, f'{worker_index}_{save_cnt}.pkl')
        # if output_buffer is not a list
        if isinstance(output_buffer, dict):
            output_buffer = [output_buffer]
        if len(output_buffer) > 0:
            with open(save_path, 'wb') as f:
                pickle.dump(output_buffer, f)
            file_info = {}
            kalman_difficulty = np.stack(
                [x['kalman_difficulty'] for x in output_buffer])
            file_info['kalman_difficulty'] = kalman_difficulty
            file_info['sample_num'] = len(output_buffer)
            file_list[save_path] = file_info

        return file_list

    def preprocess(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        The preprocess method takes in the simulation_json dataset that looks
        like the following:

[
    {
        "time_step": 0.0,
        "ego": [
            1.8119741253874913, x 
            8.637991404456686, y 
            -10.695767092600466, z
            -15.85606934102197, phi
            9.912715928234958, theta
            119.69886507984286, psi
            20.563904932489752 velocity
        ],
        "controls": [
            -15.85606934102197,
            9.912715928234958,
            119.69886507984286,
            20.563904932489752
        ],
        "vehicles": [
            [
                81.87859958969726,
                209.59954128534076,
                -11.465522178831698,
                38.27360595832225,
                8.33842486077979,
                180.07239375908537,
                23.234717016016347
            ],
            [
                48.99570030328157,
                -158.87298696652576,
                -7.894432613039017,
                -10.584679430173468,
                -15.946805538295175,
                192.81276618599486,
                23.267931222037586
            ],
            [
                58.21092978309681,
                118.05387049092073,
                -10.543774849551916,
                -16.85697854395724,
                2.7882047624047592,
                201.01239203215022,
                20.723647500169015
            ]
        ]
        }, ...

        At the end returns a dictionary with the following keys:
        - sdc_track_index: int: the index of the self-driving car in the dataset
        - timestamps_seconds: Array[float]: the timestamps of the simulation
        - track_infos: Dict[str, Any]: the information of the tracks in the simulation
            - object_type: List[int]: the type of the object
            - trajs: np.array:[num_objects, num_timestamps, num_attrs]: the trajectory of the object
                - Where num_attributes for the UAV is [x, y, z, phi, theta, psi, velocity, radius]
        """
        tracks: Dict[str, Any] = scenario['tracks']
        past_length: int = self.config['past_length']
        future_length: int = self.config['future_length']
        total_steps: int = past_length + future_length
        starting_frame: int = self.starting_frame
        ending_frame: int = starting_frame + total_steps
        trajectory_sample_interval: int = self.config['trajectory_sample_interval']
        frequency_mask: np.array = generate_mask(
            past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = {
            # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_id': [],
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items():
            pass

    def process(self, internal_format: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        This is where we will process the data into past and futrue trajectories
        """
        processed_data: List[Dict[str, Any]] = []

        for sim_data in internal_format:
            segment_data: List[np.array] = sim_data['segment_data']
            segment_idx: List[int] = sim_data['segmented_idx']
            object_type: List[int] = sim_data['object_type']
            timestamps: List[float] = sim_data['timestamp']
            idx_to_track: int = sim_data['idx_to_track']

            # track_index_to_predict = np.array(
            #     idx_to_track['tracks_to_predict']['track_index'])
            processed_sim: List[Dict[str, Any]] = []
            for i, segment in enumerate(segment_data):
                # let's keep this simple all centered objects are the vehicles
                processed_traj: Dict[str, Any] = {
                    'center_objects': [],
                    'center_objects_type': [],
                    'center_objects_past': [],
                    'center_objects_future': [],
                }

                obj_trajs_full: np.array = segment
                obj_trajs_full[:, :, HEADING_IDX] = np.deg2rad(
                    obj_trajs_full[:, :, HEADING_IDX])
                current_time_index: int = segment_idx[i]
                obj_types: List[int] = object_type
                obj_trajs_past: np.array = obj_trajs_full[:, :self.past_len, :]
                obj_trajs_future: np.array = obj_trajs_full[:,
                                                            self.past_len:, :]
                ego_traj_overall: np.array = obj_trajs_full[idx_to_track, :, :]

                # # this is basically the ego agent
                # processed_traj['center_objects'] = ego_traj_overall

                track_idx_to_predict = [i for i in range(len(obj_trajs_full))]
                center_objects = obj_trajs_full
                (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
                 obj_trajs_future_mask, center_gt_trajs,
                 center_gt_trajs_mask, center_gt_final_valid_idx,
                 track_index_to_predict_new) = self.get_agent_data(
                    center_objects=center_objects,
                    obj_trajs_past=obj_trajs_past,
                    obj_trajs_future=obj_trajs_future,
                    track_index_to_predict=track_idx_to_predict,
                    sdc_track_index=idx_to_track,
                    timestamps=timestamps[i], obj_types=obj_types
                )

                ret_dict = {
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
                    'center_gt_trajs_src': obj_trajs_full[track_idx_to_predict]
                }

                processed_sim.append(ret_dict)
                processed_data.append(ret_dict)
            # processed_data.append(processed_sim)

        return processed_data

    def postprocess(self, output):
        pass

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
        center_objects = obj_trajs_future
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )

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

        max_num_agents = self.config['max_num_agents']
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
        return obj_trajs

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.process_data[index]

    def collate_fn(self, data_list):
        """
        This function is used to collate the data list into a batch.
        The batch is a dictionary that contains the following keys:
        - batch_size: This is technically the number of objects in one single
        instance of the simulation: ie if there are 4 vehicles in the simulation 
        that we care about then the batch_size is 4
        - input_dict: a dictionary that contains the input data
        - batch_sample_count: the number of samples in the batch, should 
        be the same as the batch_size

        """
        _, num_vehicles, past_len, num_attributes = data_list[0]['obj_trajs'].shape
        # for data in data_list:
        #     data: Dict[str, Any]
        #     _, num_vehicles, past_len, num_attributes = data['obj_trajs'].shape
        input_dict = {}
        for key in data_list[0].keys():
            input_dict[key] = torch.from_numpy(
                np.stack([data[key] for data in data_list]))

        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()
        # batch_size = num_vehicles

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


class LazyBaseDataset(Dataset):
    def __init__(self, config: Dict[str, Any], 
                 is_test:bool = False,
                 is_validation: bool = False, 
                 num_samples: int = None):
        """
        Initializes the dataset in lazy mode.
        Instead of loading all JSON files and processing every segment up front,
        we build an index mapping that treats each segment (slice) as a separate sample.
        """
        self.is_validation = is_validation
        self.config = config
        # Determine the data directory based on whether we're validating or training.
        #self.data_path = config['val_data_path'] if is_validation else config['train_data_path']
        if is_test:
            print("Loading test data")
            self.data_path = config['test_data_path']
        elif is_validation:
            self.data_path = config['val_data_path']
        else:             
            self.data_path: str = config['train_data_path']

        
        # List all JSON files.
        self.json_files: List[str] = glob.glob(os.path.join(self.data_path, "*.json"))
        if num_samples is not None:
            self.json_files = self.json_files[:num_samples]
            
        self.past_len = config['past_len']
        self.future_len = config['future_len']
        # Step size for sliding window segmentation (default: 1).
        self.step_size = config.get('step_size', 1)
        
        # Build an index mapping from global segment index to a tuple (file_index, local_segment_index)
        self.index_map: List[Tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.json_files):
            # Open each file and count the timesteps.
            with open(file_path, 'r') as f:
                sim_data = json.load(f)
            total_steps = len(sim_data)  # Assuming sim_data is a list of timesteps.
            total_len = self.past_len + self.future_len

            # Determine how many segments can be extracted.
            # The segmentation loop will run from start_idx = total_len to (len(sim_data) - total_len)
            #num_segments = max(0, (total_steps - 2 * total_len + self.step_size) // self.step_size)
            num_segments = self.compute_num_segments(total_steps, total_len, self.step_size)
            for local_seg_idx in range(num_segments):
                self.index_map.append((file_idx, local_seg_idx))
                
        print(f"Initialized dataset with {len(self.index_map)} segments across {len(self.json_files)} files.")

    def __len__(self):
        # The length is now the total number of segments.
        return len(self.index_map)
    
    def __getitem__(self, global_index: int) -> Dict[str, Any]:
        """
        Maps the global segment index to the appropriate JSON file and local segment index.
        Loads the file, processes its segments, and returns the requested segment.
        """
        if global_index >= len(self.index_map):
            raise IndexError(f"Index {global_index} out of bounds.")
        
        file_idx, local_seg_idx = self.index_map[global_index]
        file_path = self.json_files[file_idx]
        # Load and process the file (all segments) on demand.
        segments = self.load_and_process_file(file_path)
        # Return the segment corresponding to the local index.
         
        return segments[local_seg_idx]

    def compute_num_segments(self, total_steps: int, total_len: int, step_size: int) -> int:
        # This uses the same range logic as in load_and_process_file.
        return len(range(total_len, total_steps - total_len + 1, step_size))


    def load_and_process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads one JSON file, converts the raw JSON into numpy arrays, segments the data into overlapping
        chunks, and then processes each segment.
        """
        with open(file_path, 'r') as f:
            sim_data: List[Dict[str, Any]] = json.load(f)
        
        # Build raw arrays from the JSON.
        overall_ego_position: List[List[float]] = []
        overall_controls: List[List[float]] = []
        overall_timestamps: List[float] = []
        num_other_vehicles: int = len(sim_data[0]['vehicles'])
        overall_pursuer_positions: List[List[Any]] = [[] for _ in range(num_other_vehicles)]
        
        for current_info in sim_data:
            overall_ego_position.append(current_info['ego'])
            overall_controls.append(current_info['controls'])
            overall_timestamps.append(current_info['time_step'])
            for j, pursuer_info in enumerate(current_info['vehicles']):
                overall_pursuer_positions[j].append(pursuer_info)
                
        overall_ego_position = np.array(overall_ego_position)
        overall_controls = np.array(overall_controls)
        for i, veh in enumerate(overall_pursuer_positions):
            overall_pursuer_positions[i] = np.array(veh)
        overall_pursuer_positions = np.stack(overall_pursuer_positions)
        # Combine ego and pursuer trajectories into one array.
        overall_traj: np.array = np.vstack([overall_ego_position[np.newaxis, ...], overall_pursuer_positions])
        
        # Segment the trajectory into overlapping chunks.
        total_len = self.past_len + self.future_len
        segments: List[Dict[str, Any]] = []
        idx_counter = 0
        # The segmentation loop should match the logic used when building the index_map.
        # for start_idx in range(total_len, len(sim_data) - total_len, self.step_size):
        for start_idx in range(total_len, len(sim_data) - total_len + 1, self.step_size):
            segment = overall_traj[:, start_idx - total_len:start_idx, :]
            processed_segment = self.process_segment(segment, overall_timestamps[start_idx:start_idx + total_len], idx_counter)
            segments.append(processed_segment)
            idx_counter += 1
        return segments

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

        max_num_agents = self.config['max_num_agents']
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

    def process(self, sim_data: Dict[str,Any]) -> Dict[str,Any]:
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
        position_noise:float = 0.5
        obj_trajs_past[:, :, 0:2] += np.random.normal(0, position_noise, obj_trajs_past[:, :, 0:2].shape)  # (Mean 0, Std 0.1m)
    
        # Multiplicative noise (simulating sensor drift)
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

    def collate_fn(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function (if using PyTorch DataLoader).
        This example stacks the segment_data from each sample.
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
        
        # segment_data_list = [sample['segment_data'] for sample in data_list]
        # input_dict = {'segment_data': torch.from_numpy(np.stack(segment_data_list))}
        # return {'batch_size': len(data_list), 'input_dict': input_dict}
        return batch_dict
    
    
    def infer_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data in internal format and return the processed data.
        The data must be in the same format as how the JSON was configured
            "time_step": 0.0,
            "ego": [
                -7.613882410665941,
                -5.466825917862023,
                60.80907589965785,
                18.203423362399416,
                -18.712367314401995,
                122.46359980956773,
                15.851284408700968
            ],
            "controls": [
                0.0,
                -18.712367314401995,
                122.46359980956773,
                15.851284408700968
            ],
            "vehicles": [
                [
                    -368.92067515595505,
                    48.05627804498421,
                    79.029477950206,
                    -27.425694614590324,
                    -9.060225535385854,
                    0.3235721558229639,
                    30.70281163597849
                ],
                [
                    -242.1243925775696,
                    -227.65438057680035,
                    53.66965889643544,
                    -19.838792625897646,
                    4.513907270243771,
                    26.216071025024867,
                    25.34648769611909
                ]
            ]
        }
        """
        
        return self.process(data)


    def transform_with_current_heading(self, 
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
    

    def rotate_predicted_trajs_by_heading(self, pred_trajs, heading_index, x_y_indices=(0,1)):
        """
        Rotates the (x,y) coordinates of predicted trajectories using their own predicted heading.
        
        This function is similar in structure to your inverse_transform_trajs_from_center_coords,
        but here the rotation angle is taken from the predicted heading in the trajectories themselves.
        
        Args:
            pred_trajs (np.ndarray): Predicted trajectories of shape (B, M, T, F)
                where B = number of agents,
                    M = number of modes,
                    T = number of timesteps,
                    F = number of features.
            heading_index (int): Index of the predicted heading within the feature dimension.
            x_y_indices (tuple): Tuple of indices for the x and y coordinates (default (0,1)).
            
        Returns:
            np.ndarray: Predicted trajectories with rotated (x,y) coordinates.
        """
        B, M, T, F = pred_trajs.shape

        # --- Extract x,y and predicted heading ---
        # x,y coordinates shape: (B, M, T, 2)
        points_xy = pred_trajs[..., x_y_indices[0]:x_y_indices[1]+1]
        # Predicted heading for each agent, mode, and timestep: shape (B, M, T)
        pred_heading = pred_trajs[..., heading_index]
        
        # --- Build rotation matrices for each agent, mode, and timestep ---
        # Standard rotation matrix for an angle h (counterclockwise):
        # R(h) = [ cos(h)  -sin(h) ]
        #        [ sin(h)   cos(h) ]
        cos_vals = np.cos(pred_heading)  # shape: (B, M, T)
        sin_vals = np.sin(pred_heading)  # shape: (B, M, T)
        
        # Build a rotation matrix R for each (B, M, T)
        R = np.empty((B, M, T, 2, 2), dtype=pred_trajs.dtype)
        R[..., 0, 0] = cos_vals
        R[..., 0, 1] = -sin_vals
        R[..., 1, 0] = sin_vals
        R[..., 1, 1] = cos_vals

        # --- Apply the rotation ---
        # Use np.einsum to multiply the rotation matrix with the (x,y) vectors:
        # The result is rotated_xy with shape (B, M, T, 2).
        rotated_xy = np.einsum('bmtij,bmtj->bmt i', R, points_xy)
        
        # --- Create the output ---
        # Make a copy of the predicted trajectories and replace the (x,y) coordinates.
        rotated_trajs = pred_trajs.copy()
        rotated_trajs[..., x_y_indices[0]:x_y_indices[1]+1] = rotated_xy

        return rotated_trajs
    
    
class LSTMDataset(Dataset):
    def __init__(self, config: Dict[str, Any], 
                 is_test:bool = False,
                 is_validation: bool = False, 
                 num_samples: int = None):
        """
        Initializes the dataset in lazy mode.
        Instead of loading all JSON files and processing every segment up front,
        we build an index mapping that treats each segment (slice) as a separate sample.
        """
        self.is_validation = is_validation
        self.config = config
        # Determine the data directory based on whether we're validating or training.
        #self.data_path = config['val_data_path'] if is_validation else config['train_data_path']
        if is_test:
            print("Loading test data")
            self.data_path = config['test_data_path']
        elif is_validation:
            self.data_path = config['val_data_path']
        else:             
            self.data_path: str = config['train_data_path']

        
        # List all JSON files.
        self.json_files: List[str] = glob.glob(os.path.join(self.data_path, "*.json"))
        if num_samples is not None:
            self.json_files = self.json_files[:num_samples]
            
        self.past_len = config['past_len']
        self.future_len = config['future_len']
        # Step size for sliding window segmentation (default: 1).
        self.step_size = config.get('step_size', 1)
        
        # Build an index mapping from global segment index to a tuple (file_index, local_segment_index)
        self.index_map: List[Tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.json_files):
            # Open each file and count the timesteps.
            with open(file_path, 'r') as f:
                sim_data = json.load(f)
            total_steps = len(sim_data)  # Assuming sim_data is a list of timesteps.
            total_len = self.past_len + self.future_len

            # Determine how many segments can be extracted.
            # The segmentation loop will run from start_idx = total_len to (len(sim_data) - total_len)
            #num_segments = max(0, (total_steps - 2 * total_len + self.step_size) // self.step_size)
            num_segments = self.compute_num_segments(total_steps, total_len, self.step_size)
            for local_seg_idx in range(num_segments):
                self.index_map.append((file_idx, local_seg_idx))
                
        print(f"Initialized dataset with {len(self.index_map)} segments across {len(self.json_files)} files.")

    def __len__(self):
        # The length is now the total number of segments.
        return len(self.index_map)
    
    def __getitem__(self, global_index: int) -> Dict[str, Any]:
        """
        Maps the global segment index to the appropriate JSON file and local segment index.
        Loads the file, processes its segments, and returns the requested segment.
        """
        if global_index >= len(self.index_map):
            raise IndexError(f"Index {global_index} out of bounds.")
        
        file_idx, local_seg_idx = self.index_map[global_index]
        file_path = self.json_files[file_idx]
        # Load and process the file (all segments) on demand.
        segments = self.load_and_process_file(file_path)
        # Return the segment corresponding to the local index.
         
        return segments[local_seg_idx]

    def compute_num_segments(self, total_steps: int, total_len: int, step_size: int) -> int:
        # This uses the same range logic as in load_and_process_file.
        return len(range(total_len, total_steps - total_len + 1, step_size))

    def load_and_process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads one JSON file, converts the raw JSON into numpy arrays, segments the data into overlapping
        chunks, and then processes each segment.
        """
        with open(file_path, 'r') as f:
            sim_data: List[Dict[str, Any]] = json.load(f)
        
        # Build raw arrays from the JSON.
        overall_ego_position: List[List[float]] = []
        overall_controls: List[List[float]] = []
        overall_timestamps: List[float] = []
        num_other_vehicles: int = len(sim_data[0]['vehicles'])
        overall_pursuer_positions: List[List[Any]] = [[] for _ in range(num_other_vehicles)]
        
        for current_info in sim_data:
            overall_ego_position.append(current_info['ego'])
            overall_controls.append(current_info['controls'])
            overall_timestamps.append(current_info['time_step'])
            for j, pursuer_info in enumerate(current_info['vehicles']):
                overall_pursuer_positions[j].append(pursuer_info)
                
        overall_ego_position = np.array(overall_ego_position)
        overall_controls = np.array(overall_controls)
        for i, veh in enumerate(overall_pursuer_positions):
            overall_pursuer_positions[i] = np.array(veh)
        overall_pursuer_positions = np.stack(overall_pursuer_positions)
        # Combine ego and pursuer trajectories into one array.
        overall_traj: np.array = np.vstack([overall_ego_position[np.newaxis, ...], overall_pursuer_positions])
        
        # Segment the trajectory into overlapping chunks.
        total_len = self.past_len + self.future_len
        segments: List[Dict[str, Any]] = []
        idx_counter = 0
        # The segmentation loop should match the logic used when building the index_map.
        # for start_idx in range(total_len, len(sim_data) - total_len, self.step_size):
        for start_idx in range(total_len, len(sim_data) - total_len + 1, self.step_size):
            segment = overall_traj[:, start_idx - total_len:start_idx, :]
            processed_segment = self.process_segment(segment, overall_timestamps[start_idx:start_idx + total_len], idx_counter)
            segments.append(processed_segment)
            idx_counter += 1
        return segments
    
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
    
    def transform_trajs_to_center_coords(
        self, obj_trajs: np.array, 
        center_xyz: np.array, 
        center_heading: np.array, 
        heading_index: int, rot_vel_index=None):
        """
        Transforms trajectories from global coordinates to 
        center (ego-centric) coordinates.
        """
        num_center_objects = center_xyz.shape[0]
        num_objects, num_timestamps, num_attributes = obj_trajs.shape
        
        # subtract the position of the center object to zero out the position
        obj_trajs[:,:, 0:3] -= center_xyz[:, None, :]
        
        # zero out the heading of the center object as well
        obj_trajs[:,:, heading_index] -= center_heading

        return obj_trajs
    
    def get_agent_data(
        self, 
        center_objects: np.array,
        obj_trajs_past: np.array,
        obj_trajs_future: np.array,
        track_index_to_predict: List[int],
        sdc_track_index: int,
        timestamps: List[float],
        obj_types: List[int]
    ):
        """
        Centers the trajectories around the ego vehicle and returns the processed data.
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
        
        obj_trajs_future = obj_trajs_future.astype(np.float32)
        copy_obj = obj_trajs_future.copy()
        center_objects = obj_trajs_future
        
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )
        
        return (
            obj_trajs,
            obj_trajs_future,
            timestamps,
            obj_types
        )
        
    
    def process(self, sim_data: Dict[str,Any]) -> Dict[str,Any]:
        """
        Need to return a dictionary
        """
        timestamp = sim_data['timestamp']
        idx_to_track: int = sim_data['idx_to_track']
        obj_trajs_full: np.array = sim_data['segment_data']
        obj_trajs_full[:, :, HEADING_IDX] = np.deg2rad(
            obj_trajs_full[:, :, HEADING_IDX])
        obj_types: List[int] = sim_data['object_type']
        obj_trajs_past: np.array = obj_trajs_full[:, :self.past_len, :]
        obj_trajs_future: np.array = obj_trajs_full[:,
                                                    self.past_len:, :]
        
        track_idx_to_predict = [i for i in range(len(obj_trajs_full))]
        center_objects = obj_trajs_full.copy()
        original_pos_past: np.array = obj_trajs_past.copy()
        
        # sanity check let's plot the trajectories
        # fig, ax = plt.subplots()
        # for i in range(obj_trajs_past.shape[0]):
        #     ax.plot(obj_trajs_past[i, :, 0], obj_trajs_past[i, :, 1], label=f"Agent {i}")
        # ax.legend()
        # # title 
        # ax.set_title("Trajectories Uncentered")
    
        # ========================== NOISE INJECTION ==========================

        ## 1. Measurement Noise (Sensor Errors)
        # Gaussian position noise (simulating GPS or LIDAR errors)
        position_noise:float = 0.5
        obj_trajs_past[:, :, 0:2] += np.random.normal(0, 
                                                      position_noise, 
                                                      obj_trajs_past[:, :, 0:2].shape)  # (Mean 0, Std 0.1m)
    
        # Multiplicative noise (simulating sensor drift)
        obj_trajs_past[:, :, 0:2] *= np.random.normal(1, 
                                                      0.02, 
                                                      obj_trajs_past[:, :, 0:2].shape)  # 2% variation

        # ## 2. Process Noise (Motion Model Uncertainty)
        # # Random walk noise (simulating object drift over time)
        # # drift = np.cumsum(np.random.normal(0, 0.05, obj_trajs_past[:, :, 0:2].shape), axis=1)  # Accumulate small movements
        # # obj_trajs_past[:, :, 0:2] += drift

        # # Velocity noise (simulating varying acceleration)
        # velocity_noise = np.random.normal(0, 0.2, obj_trajs_past[:, :, VELOCITY_IDX].shape)  # Velocity in (x, y)
        # obj_trajs_past[:, :, VELOCITY_IDX] += velocity_noise
        
        (obj_trajs_data, obj_trajs_future, timestamps, obj_types) = self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_idx_to_predict,
            sdc_track_index=idx_to_track,
            timestamps=timestamp, obj_types=obj_types
        )
        
        ret: Dict[str, Any] = {
            # 'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data, # this is the past trajectory
            'center_objects_world': center_objects,
            'center_objects_type': np.array(obj_types),
            'center_gt_trajs': obj_trajs_future,
            'original_pos_past': original_pos_past,
        }

        return ret

    def collate_fn(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
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

class SingleLSTMDataset(Dataset):
    def __init__(self, config: Dict[str, Any], 
                 is_test: bool = False,
                 is_validation: bool = False, 
                 num_samples: int = None):
        """
        Initializes the dataset in lazy mode.
        Each segment corresponds to a slice of the ego agent trajectory.
        """
        self.is_validation = is_validation
        self.config = config
        
        if is_test:
            print("Loading test data")
            self.data_path = config['test_data_path']
        elif is_validation:
            self.data_path = config['val_data_path']
        else:             
            self.data_path = config['train_data_path']

        # List all JSON files.
        self.json_files: List[str] = glob.glob(os.path.join(self.data_path, "*.json"))
        if num_samples is not None:
            self.json_files = self.json_files[:num_samples]
            
        self.past_len = config['past_len']
        self.future_len = config['future_len']
        self.step_size = config.get('step_size', 1)
        
        # Build an index mapping from global segment index to (file_index, local_segment_index)
        self.index_map: List[Tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.json_files):
            with open(file_path, 'r') as f:
                sim_data = json.load(f)
            total_steps = len(sim_data)  # assuming sim_data is a list of timesteps
            total_len = self.past_len + self.future_len
            num_segments = self.compute_num_segments(total_steps, total_len, self.step_size)
            for local_seg_idx in range(num_segments):
                self.index_map.append((file_idx, local_seg_idx))
                
        print(f"Initialized dataset with {len(self.index_map)} segments across {len(self.json_files)} files.")

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, global_index: int) -> Dict[str, Any]:
        if global_index >= len(self.index_map):
            raise IndexError(f"Index {global_index} out of bounds.")
        
        file_idx, local_seg_idx = self.index_map[global_index]
        file_path = self.json_files[file_idx]
        segments = self.load_and_process_file(file_path)
        return segments[local_seg_idx]

    def compute_num_segments(self, total_steps: int, total_len: int, step_size: int) -> int:
        return len(range(total_len, total_steps - total_len + 1, step_size))

    def load_and_process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a JSON file and extracts only the ego trajectory.
        """
        with open(file_path, 'r') as f:
            sim_data: List[Dict[str, Any]] = json.load(f)
        
        # Build raw arrays from the JSON (only ego data is used).
        overall_ego_position: List[List[float]] = []
        overall_controls: List[List[float]] = []
        overall_timestamps: List[float] = []
        
        for current_info in sim_data:
            overall_ego_position.append(current_info['ego'])
            overall_controls.append(current_info['controls'])
            overall_timestamps.append(current_info['time_step'])
            
        overall_ego_position = np.array(overall_ego_position)
        overall_controls = np.array(overall_controls)
        # Use only the ego trajectory (add a new axis so the agent dimension is 1).
        overall_traj: np.array = overall_ego_position[np.newaxis, ...]  # shape: (1, T, features)
        
        total_len = self.past_len + self.future_len
        segments: List[Dict[str, Any]] = []
        idx_counter = 0
        for start_idx in range(total_len, len(sim_data) - total_len + 1, self.step_size):
            # Slice the trajectory for the segment.
            segment = overall_traj[:, start_idx - total_len:start_idx, :]
            processed_segment = self.process_segment(segment, overall_timestamps[start_idx:start_idx + total_len], idx_counter)
            segments.append(processed_segment)
            idx_counter += 1
        return segments
    
    def process_segment(self, segment: np.array, timestamps: List[float], idx: int) -> Dict[str, Any]:
        """
        Processes a segment to output only the ego agent.
        Assumes that the ego trajectory is the first (and only) row.
        """
        # Select only the ego agent (index 0) and keep the agent dimension.
        ego_segment = segment[0:1, :, :]  # shape: (1, total_steps, features)
        
        processed = {
            'object_type': [VEHICLE],  # assuming VEHICLE is defined
            'timestamp': timestamps,
            'idx_to_track': 0,
            'segment_idx': idx,
            'segment_data': ego_segment,
        }
        
        tracks_to_predict = {
            'track_index': [0],
            'object_type': [VEHICLE]
        }
        processed['tracks_to_predict'] = tracks_to_predict
                
        return processed
    
    def transform_trajs_to_center_coords(
        self, obj_trajs: np.array, 
        center_xyz: np.array, 
        center_heading: np.array, 
        heading_index: int, rot_vel_index=None):
        """
        Transforms trajectories from global to ego-centric coordinates.
        """
        num_center_objects = center_xyz.shape[0]
        num_objects, num_timestamps, num_attributes = obj_trajs.shape
        
        obj_trajs[:, :, 0:3] -= center_xyz[:, None, :]
        obj_trajs[:, :, heading_index] -= center_heading

        return obj_trajs
    
    def get_agent_data(
        self, 
        center_objects: np.array,
        obj_trajs_past: np.array,
        obj_trajs_future: np.array,
        track_index_to_predict: List[int],
        sdc_track_index: int,
        timestamps: List[float],
        obj_types: List[int]
    ):
        """
        Centers trajectories around the ego vehicle and returns the processed data.
        """
        center_objects = obj_trajs_past
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )
        
        obj_trajs_future = obj_trajs_future.astype(np.float32)
        copy_obj = obj_trajs_future.copy()
        center_objects = obj_trajs_future
        
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0, 0:3],
            center_heading=center_objects[:, :, HEADING_IDX],
            heading_index=HEADING_IDX, rot_vel_index=[7, 8]
        )
        
        return (
            obj_trajs,
            obj_trajs_future,
            timestamps,
            obj_types
        )
        
    def process(self, sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single segment and returns the dictionary.
        """
        timestamp = sim_data['timestamp']
        idx_to_track: int = sim_data['idx_to_track']
        obj_trajs_full: np.array = sim_data['segment_data']
        # Convert heading angles from degrees to radians (only for ego).
        obj_trajs_full[:, :, HEADING_IDX] = np.deg2rad(obj_trajs_full[:, :, HEADING_IDX])
        obj_types: List[int] = sim_data['object_type']
        obj_trajs_past: np.array = obj_trajs_full[:, :self.past_len, :]
        obj_trajs_future: np.array = obj_trajs_full[:, self.past_len:, :]
        
        track_idx_to_predict = [0]
        center_objects = obj_trajs_full.copy()
        original_pos_past = obj_trajs_past.copy()
        
        (obj_trajs_data, obj_trajs_future, timestamps, obj_types) = self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_idx_to_predict,
            sdc_track_index=idx_to_track,
            timestamps=timestamp, 
            obj_types=obj_types
        )
        
        ret: Dict[str, Any] = {
            'obj_trajs': obj_trajs_data,  # past trajectory for ego
            'center_objects_world': center_objects,
            'center_objects_type': np.array(obj_types),
            'center_gt_trajs': obj_trajs_future,  # future trajectory for ego
            'original_pos_past': original_pos_past,
        }

        return ret

    def collate_fn(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for the DataLoader.
        """
        processed_data: List[Dict[str, Any]] = [self.process(sample) for sample in data_list]
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
