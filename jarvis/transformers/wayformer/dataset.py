from typing import Dict, Any, List
from omegaconf import OmegaConf
import hydra
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

# TODO: Reclass as Enum
VEHICLE = 1
HEADING_IDX = 5
VELOCITY_IDX = 6


def rotate_points_along_z(points, angle):
    """
    Rotate points around the Z-axis using the given angle.

    Args:
        points: ndarray of shape (B, N, 3 + C) - B batches, N points per batch, 3 coordinates (x, y, z) + C extra channels
        angle: ndarray of shape (B,) - angles for each batch in radians

    Returns:
        Rotated points as an ndarray.
    """

    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == 2

    # Cosine and sine of the angles
    cosa = np.cos(angle)
    sina = np.sin(angle)

    # if is_2d:
    # Rotation matrix for 2D
    rot_matrix = np.stack((
        cosa, sina,
        -sina, cosa
    ), axis=1).reshape(-1, 2, 2)

    # Apply rotation
    # points_rot = np.matmul(points, rot_matrix)
    points_rot = np.einsum('bnj,nji->bni', points, rot_matrix)

    return points_rot


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
                 is_validation=False,
                 num_samples=None):

        self.is_validation: bool = is_validation

        if is_validation:
            self.data_path: str = config['val_data_path']
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

            if idx_counter == 2:
                break

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

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        pass

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
        # original_obj_trajs = obj_trajs.copy()
        # # TODO: REMOVE
        # test = original_obj_trajs[:, :, 0:3] - center_xyz[:, None, :]
        # obj_trajs = np.tile(
        #     obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        # obj_trajs[:, :, :, 0:center_xyz.shape[1]
        #           ] -= center_xyz[:, None, None, :]
        # obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
        #     points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        #     angle=-center_heading
        # ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for i in range(num_center_objects):
        #     x = test[i, :, 0]
        #     y = test[i, :, 1]
        #     ax.plot(x, y, label=f'center_{i}')
        # ax.legend()

        # fig, ax = plt.subplots()
        # for i in range(num_center_objects):
        #     x = obj_trajs[i, :, 0]
        #     y = obj_trajs[i, :, 1]
        #     ax.plot(x, y, label=f'center_{i}')
        # ax.legend()
        # plt.show()

        # plot 3d
        # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        # for i in range(num_center_objects):
        #     x = test[i, :, 0]
        #     y = test[i, :, 1]
        #     z = test[i, :, 2]
        #     ax.plot(x, y, z, label=f'center_{i}')
        # # title
        # ax.set_title('Zero')

        # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        # for i in range(num_center_objects):
        #     x = obj_trajs[i, :, 0]
        #     y = obj_trajs[i, :, 1]
        #     z = obj_trajs[i, :, 2]
        #     ax.plot(x, y, z, label=f'center_{i}')

        # plt.show()

        # TODO: ADD ROTATION correctly based on the heading
        obj_trajs = np.tile(
            obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]
                  ] -= center_xyz[:, None, None, :]
        # obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
        #     points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        #     angle=-center_heading
        # ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        # obj_trajs[:, :, :, heading_index] -= center_heading[:, None, :]

        # rotate direction of velocity
        # if rot_vel_index is not None:
        #     assert len(rot_vel_index) == 2
        #     obj_trajs[:, :, :, rot_vel_index] = rotate_points_along_z(
        #         points=obj_trajs[:, :, :, rot_vel_index].reshape(
        #             num_center_objects, -1, 2),
        #         angle=-center_heading
        #     ).reshape(num_center_objects, num_objects, num_timestamps, 2)

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

        # batch_size = num_vehicles

        # batch_list = []
        # for batch in data_list:
        #     batch_list += batch

        # batch_size = len(batch_list)
        # key_to_list = {}
        # for key in batch_list[0].keys():
        #     key_to_list[key] = [batch_list[bs_idx][key]
        #                         for bs_idx in range(batch_size)]

        # input_dict = {}
        # for key, val_list in key_to_list.items():
        #     # if val_list is str:
        #     try:
        #         input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
        #     except:
        #         input_dict[key] = val_list

        # input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        # batch_dict = {'batch_size': batch_size,
        #               'input_dict': input_dict, 'batch_sample_count': batch_size}
        return batch_dict
