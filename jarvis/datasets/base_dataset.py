from torch.utils.data import Dataset
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json
import ijson
import concurrent.futures
import glob
import os
import torch
import time
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any, Optional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter
from jarvis.datasets.common_utils import generate_mask
from collections import defaultdict
from einops import rearrange

"""
https://medium.com/geekculture/pytorch-datasets-dataloader-samplers-and-the-collat-fn-bbfc7c527cf1
"""

# make an Enum for the different types of datasets
from enum import Enum


class DatasetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ObjectTypes(Enum):
    UNSET = 0
    EGO = 1
    PURSUERS = 2
    TARGET = 3


class StateIndex(Enum):
    X = 0
    Y = 1
    YAW = 2
    VEL = 3
    VX = 4
    VY = 5


default_value = 0
# object_type = defaultdict(lambda: default_value, ObjectTypes)


class BaseDataset(Dataset):
    """
    Base class for creating custom datasets.
    Specify the training and validation datasets using this class
    """

    def __init__(self, config: Dict[str, Any],
                 is_validation: bool = False) -> None:
        super().__init__()
        self.is_validation: bool = is_validation
        if is_validation:
            self.data_path: str = config['val_data_path']
        else:
            self.data_path: str = config['train_data_path']

        self.config: Dict[str, Any] = config
        self.past_length: int = config['past_len']
        self.future_length: int = config['future_len']
        self.total_length: int = self.past_length + self.future_length
        self.trajectory_sample_interval: int = config['trajectory_sample_interval']
        self.data_loaded: Dict[str, Any] = {}

        self.num_samples: int = config['num_samples']
        self.data_loaded: Dict[str, Any] = {}
        self.final_data: Dict[List[Dict[str, Any]]] = {}
        self.overall_data: List = []
        self.load_data()

    def split_individual_json(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process individual JSON files

        If the data input is too large we'll split it into smaller chunks

        In each small chunk is length of total_length
        The data will be a dictionary with the following
        {
            'ego': List[Dict[str, Any]],
            'vehicle_0': List[Dict[str, Any]],
            'vehicle_n': List[Dict[str, Any]],
            'time_step': List[Dict[str, Any]]
        }

        """
        data_lists: List[List[Dict[str, Any]]] = []

        # check the length of the data
        if len(data) >= self.total_length:
            # how many times can we split the data
            num_splits: int = len(data) // self.total_length
            # split the data into smaller chunks
            # Slicing throuhg 0 to num_splits
            for i in range(num_splits):
                data_lists.append(
                    data[i * self.total_length: (i + 1) * self.total_length])

        info_lists: List[Dict[str, Any]] = []
        for split in data_lists[:-1]:
            if len(split) != self.total_length:
                raise ValueError(
                    f"Data length {len(split)} does not match total length {self.total_length}")
            tracks = {
                "ego": [],
                "time": []
            }

            for i, s in enumerate(split):
                # check if time_step is in the dictionary
                if 'ego' in s:
                    tracks["ego"].append(s['ego'])
                if 'vehicles' in s:
                    for i, vehicle in enumerate(s['vehicles']):
                        veh_name = f"vehicle_{i}"
                        if veh_name not in tracks:
                            tracks[veh_name] = []
                        tracks[veh_name].append(vehicle)
                if 'time_step' in s:
                    tracks["time"].append(s['time_step'])

            info_lists.append(tracks)
        return info_lists
    # Helper function to load and process each JSON file

    def load_and_process_file(self, data_file: str) -> (str, List[Dict[str, Any]]):
        with open(data_file) as f:
            data = json.load(f)
        # Process the data
        processed_data = self.split_individual_json(data)
        return data_file, processed_data

    def load_data_in_parallel(self, json_files: List[str], max_workers: int = 4):
        """
        This will return a dictionary of the loaded data with a format of the following:
        {
            'sim_number_1.json': List[Dict[str, Any]],
            'sim_number_2.json': List[Dict[str, Any]],
        }
        Where in each key the value is a list of dictionaries containing the data
        [Dictionary from 0 to total_length, Dictionary from next total_length to 2*total_length]
        ie.
        Total length is found from the config file
        [Dictionary from 0 to 81, 82 to 162, 163 to 243, ...]
        """
        self.data_loaded = {}
        # Use ThreadPoolExecutor for I/O-bound tasks, or ProcessPoolExecutor if CPU-bound
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each file to be processed in parallel
            future_to_file = {executor.submit(
                self.load_and_process_file, data_file): data_file for data_file in json_files}

            for future in concurrent.futures.as_completed(future_to_file):
                data_file = future_to_file[future]
                try:
                    data_file, data = future.result()  # Retrieve results
                    self.data_loaded[data_file] = data
                except Exception as e:
                    print(f"Error processing {data_file}: {e}")

    def load_data(self) -> None:
        """
        Load the dataset from the specified path
        we want to get the dataset
        #TODO: Process the data
        """
        if self.is_validation:
            print("Loading validation data...")
        else:
            print("Loading training data...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Path {self.data_path} does not exist")

        # Use glob to get all JSON files
        json_files: List[str] = glob.glob(
            os.path.join(self.data_path, "*.json"))

        if self.num_samples is not None:
            json_files: List[str] = json_files[:self.num_samples]

        start_time: float = time.time()
        self.load_data_in_parallel(json_files, max_workers=2)
        print(
            f"Loaded {len(json_files)} files in {time.time() - start_time:.2f} seconds")

        for i, (k, sample_batches) in enumerate(self.data_loaded.items()):
            self.final_data[k] = []
            for samples in sample_batches:
                samples: Dict[str, Any]

                with open("sample.pkl", "wb") as f:
                    pickle.dump(samples, f)

                output: Dict[str, Any] = self.preprocess_data(
                    data=samples, filename=k, idx=i)

                # save this data as pickle
                with open("intermediate.pkl", "wb") as f:
                    pickle.dump(output, f)

                output = self.process_data(output)
                # save this data as pickle
                with open("output_data.pkl", "wb") as f:
                    pickle.dump(output, f)

                self.final_data[k].append(output)
                self.overall_data.append(output)

        # TODO: REMOVE THIS LINE
        # self.overall_data = [self.overall_data[0]]

    def preprocess_data(self, data: Dict[str, Any], filename: str,
                        idx: int) -> Dict[str, Any]:
        """
        Preprocess the data
        data input  will be a list of dictionaries:
        [Dictionary from 0 to total_length, Dictionary from next total_length to 2*total_length]
        ie.
        Total length is found from the config file
        [Dictionary from 0 to 81, 82 to 162, 163 to 243, ...]

        Each dictionary will contain the following:
        {
            'ego': List[Dict[str, Any]],
            'vehicle_0': List[Dict[str, Any]],
            'vehicle_n': List[Dict[str, Any]],
            'time_step': List[Dict[str, Any]]
        }

        Size of traj_infos['trajs'] should be (num_objects, total_length, num_features)

        """
        frequency_mask = generate_mask(
            self.past_length - 1, self.total_length, self.trajectory_sample_interval)

        track_infos = {
            'object_id': [],
            'object_type': [],
            'trajs': []
        }
        for k, v in data.items():
            if k == 'ego':
                object_type: Enum = ObjectTypes.EGO
                combined_state: np.array = np.array(v)
                # combined_state[:, 0:2] -= combined_state[0, 0:2]
                # combined_state = np.concatenate(
                #     combined_state, axis=-1
                # )
                track_infos['object_id'].append(k)
                track_infos['object_type'].append(object_type)
                track_infos['trajs'].append(combined_state)
            elif k == 'time':
                pass
                # print("Time", v)
            else:
                object_type: Enum = ObjectTypes.PURSUERS
                combined_state: np.array = np.array(v)
                # combined_state = np.concatenate(
                #     combined_state, axis=-1
                # )
                track_infos['object_id'].append(k)
                track_infos['object_type'].append(object_type)
                track_infos['trajs'].append(combined_state)

        # Stack trajectory information and apply frequency mask
        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]

        # Define the tracks_to_predict based on specific criteria
        tracks_to_predict: Dict[str, List] = {
            'track_index': [i for i, obj_id in enumerate(track_infos['object_id'])
                            if obj_id == 'ego' or "vehicle" in obj_id],
            # Set difficulty level for each
            'difficulty': [0] * len(track_infos['object_id']),
            'object_type': [track_infos['object_type'][i] \
                            for i in range(len(track_infos['object_id']))]
        }

        # Compile processed data with metadata
        name_info = f"{filename}_{idx}"
        ret = {
            'name': name_info,
            'track_info': track_infos,
            'tracks_to_predict': tracks_to_predict,
            'time': data['time'],
            'current_time_index': self.past_length - 1,
            'current_time': data['time'][0],
            'future_time': data['time'][-1]
        }

        return ret

    def get_interested_agents(self,
                              track_index_to_predict: List[int],
                              obj_trajs_full: np.ndarray,
                              current_time_index: int,
                              obj_types: List[int],
                              scene_id: str
                              ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Filters and selects agents from the given data that are valid and of a specified type at a particular time step.

        Parameters:
            track_index_to_predict (List[int]): Indices of agents to consider for prediction.
            obj_trajs_full (np.ndarray): Full trajectory data for all agents, with shape (num_objects, num_timesteps, num_features).
            current_time_index (int): The specific timestep index to check agent validity.
            obj_types (List[int]): List of agent type codes for each agent in `obj_trajs_full`.
            scene_id (str): Unique identifier for the scene being processed, used in error logging.

        Returns:
            Tuple[Optional[np.ndarray], List[int]]: A tuple containing:
                - `center_objects` (np.ndarray): Array of valid agents' states at `current_time_index` with shape (num_center_objects, num_features),
                - `track_index_to_predict_selected` (List[int]): List of selected agent indices.
                If no valid agents are found, returns (None, []).

        """
        center_objects_list: List[np.ndarray] = []
        track_index_to_predict_selected: List[int] = []

        # Retrieve desired object types from config and map to integer labels
        # selected_type = [object_type[x] for x in self.config['object_type']]
        selected_type = [ObjectTypes.EGO, ObjectTypes.PURSUERS]

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            # Validate agent presence at current time step
            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(
                    f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue

            # Filter agents by specified type
            if obj_types[obj_idx] not in selected_type:
                continue

            # Add agent data and index if valid and matching type
            center_objects_list.append(
                obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        # Handle case where no valid agents are found
        if len(center_objects_list) == 0:
            print(
                f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []

        # Stack selected agent states along the first dimension
        center_objects = np.stack(center_objects_list, axis=0)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict_selected

    def get_agent_data(
        self,
        center_objects: np.ndarray,
        obj_trajs_past: np.ndarray,
        obj_trajs_future: np.ndarray,
        track_index_to_predict: np.ndarray,
        sdc_track_index: int,
        timestamps: np.ndarray,
        obj_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes agent trajectory data and returns structured arrays for trajectory details, embeddings, and masks.

        Args:
            center_objects (np.ndarray): Array containing information on center objects with shape (num_center_objects, num_features).
            obj_trajs_past (np.ndarray): Past trajectories of all objects, shape (num_objects, num_timestamps, num_features).
            obj_trajs_future (np.ndarray): Future trajectories of all objects, shape (num_objects, num_timestamps, num_features).
            track_index_to_predict (np.ndarray): Indices of objects to be predicted, shape (num_center_objects,).
            sdc_track_index (int): Index of the self-driving car (SDC) in the list of objects.
            timestamps (np.ndarray): Array of timestamps, shape (num_timestamps,).
            obj_types (np.ndarray): Array of object types with shape (num_objects,).

        Returns:
            Dictionary containing the following:
                - obj_trajs_data (np.ndarray): Combined array with trajectory data, one-hot encoding, time, heading embeddings, and acceleration, shape (num_center_objects, max_num_agents, num_timestamps, num_features).
                - obj_trajs_mask (np.ndarray): Mask array indicating valid timestamps, shape (num_center_objects, max_num_agents, num_timestamps).
                - obj_trajs_pos (np.ndarray): Position data, shape (num_center_objects, max_num_agents, num_timestamps, 3).
                - obj_trajs_last_pos (np.ndarray): Last position of agents, shape (num_center_objects, max_num_agents, 3).
                - obj_trajs_future_state (np.ndarray): Future state with (x, y, vx, vy) for agents, shape (num_center_objects, max_num_agents, num_timestamps, 4).
                - obj_trajs_future_mask (np.ndarray): Mask array indicating valid future timestamps, shape (num_center_objects, max_num_agents, num_timestamps).
                - center_gt_trajs (np.ndarray): Ground truth future trajectories for center objects, shape (num_center_objects, num_timestamps, 4).
                - center_gt_trajs_mask (np.ndarray): Mask for valid future trajectory timestamps, shape (num_center_objects, num_timestamps).
                - center_gt_final_valid_idx (np.ndarray): Indices of the final valid timestamp for each center object, shape (num_center_objects,).
                - track_index_to_predict_new (np.ndarray): Updated track indices to predict for each center object, shape (num_center_objects,).

        """
        other_idx_encode: int = 3
        ego_idx_encode: int = 4

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        center_xyz = center_objects[:, 0:2]
        ego_init = center_xyz[sdc_track_index, 0:2]
        # center_xyz = center_objects[sdc_track_index, 0:2]
        # print("Center XYZ", center_xyz.shape)
        # get the first
        # the data is already made to be relative to the center object or the ego vehicle
        # Expand obj_trajs_past to have an additional dimension for num_center_objects
        # Now object_trajs will have shape (num_center_objects, num_objects, num_timestamps, feature_dim)
        obj_trajs = np.tile(
            obj_trajs_past[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]
                  ] -= center_xyz[:, None, None, :]

        # # obj_trajs = np.tile(obj_trajs_past, (num_center_objects, 1, 1, 1))
        object_onehot_mask = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1

        # object_onehot_mask[:, obj_types == 2, :, 1] = 1
        # object_onehot_mask[:, obj_types == 3, :, 2] = 1

        object_onehot_mask[np.arange(
            num_center_objects), track_index_to_predict, :, other_idx_encode] = 1
        object_onehot_mask[:, sdc_track_index, :, ego_idx_encode] = 1

        object_time_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        # we are are computing the sin and cos of the yaw
        # to make sure that the yaw is continuous
        object_heading_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(
            obj_trajs[:, :, :, StateIndex.YAW.value])
        object_heading_embedding[:, :, :, 1] = np.cos(
            obj_trajs[:, :, :, StateIndex.YAW.value])

        vel = obj_trajs[:, :, :, StateIndex.VEL.value:StateIndex.VY.value]
        vel_prev = np.roll(vel, shift=1, axis=2)
        acce = vel - vel_prev / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, StateIndex.X.value:StateIndex.YAW.value],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, StateIndex.VEL.value:StateIndex.VY.value],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        # obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = np.tile(
            obj_trajs_future[None, :, :, :], (num_center_objects, 1, 1, 1))
        # obj_trajs_future = np.tile(
        #     obj_trajs_future, (num_center_objects, 1, 1, 1))
        obj_trajs_future[:, :, :, 0:center_xyz.shape[1]
                         ] -= center_xyz[:, None, None, :]
        obj_trajs_future_state = obj_trajs_future[:, :, :, [
            StateIndex.X.value, StateIndex.Y.value, StateIndex.VX.value, StateIndex.VY.value]]  # (x, y, vx, vy)

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

        obj_trajs_pos = obj_trajs_data[:, :, :,
                                       StateIndex.X.value:StateIndex.Y.value]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros(
            (num_center_objects, num_objects, 2), dtype=np.float32)
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

        # mask out agents that are not in the topk or super far away
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

        # dump the center_gt_trajs
        with open("center_gt_trajs.pkl", "wb") as f:
            pickle.dump(center_gt_trajs, f)

        # add padding in case the number of agents is less than max_num_agents
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
        # Returning data in a structured dictionary
        return {
            "obj_trajs_data": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask.astype(bool),
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "track_index_to_predict_new": track_index_to_predict_new
        }

    def process_data(self, internal_format: Dict) -> None:
        """
        Process the loaded data
        """
        info: Dict[str, Any] = internal_format
        current_time_index: int = info['current_time_index']
        timestamps: np.array = np.array(
            info['time'][:current_time_index + 1], dtype=np.float32)
        track_infos: Dict[str, Any] = info['track_info']
        # tracks_to_predict: Dict[str, Any] = info['tracks_to_predict']
        track_index_to_predict = np.array(
            info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        # (num_objects, num_timestamp, 10)
        obj_trajs_full: np.array = track_infos['trajs']
        obj_trajs_past: np.array = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future: np.array = obj_trajs_full[:, current_time_index + 1:]

        center_objects: np.ndarray
        track_index_to_predict_selected: List[int]
        center_objects, track_index_to_predict_selected = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types,
            scene_id=info['name'])

        sample_num: int = center_objects.shape[0]

        agent_data: Dict[str, Any] = self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict_selected,
            sdc_track_index=0,
            timestamps=timestamps,
            obj_types=obj_types
        )

        # Unpacking the dictionary
        obj_trajs_data = agent_data["obj_trajs_data"]
        obj_trajs_mask = agent_data["obj_trajs_mask"]
        obj_trajs_pos = agent_data["obj_trajs_pos"]
        obj_trajs_last_pos = agent_data["obj_trajs_last_pos"]
        obj_trajs_future_state = agent_data["obj_trajs_future_state"]
        obj_trajs_future_mask = agent_data["obj_trajs_future_mask"]
        center_gt_trajs = agent_data["center_gt_trajs"]
        center_gt_trajs_mask = agent_data["center_gt_trajs_mask"]
        center_gt_final_valid_idx = agent_data["center_gt_final_valid_idx"]
        track_index_to_predict_new = agent_data["track_index_to_predict_new"]

        ret_dict = {
            'scenario_id': info['name'],
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            # used to select center-features
            'track_index_to_predict': track_index_to_predict_new,
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': agent_data["center_gt_trajs"],
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict],
            'uncentered_trajs_past': obj_trajs_full[:, :current_time_index + 1],
            'uncentered_trajs_future': obj_trajs_full[:, current_time_index + 1:],
        }

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        return ret_dict

    def __len__(self):
        # length = 0
        # for k, v in self.final_data.items():
        #     length += len(v)
        # return length
        return len(self.overall_data)

    def __getitem__(self, idx: int):
        return self.overall_data[idx]
        # return self.overall_data[0]

    def collate_fn(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to create batches from a list of data samples.
        """
        batch_size = len(data_list)
        key_to_list = {key: [sample[key] for sample in data_list]
                       for key in data_list[0].keys()}

        input_dict = {}
        for key, val_list in key_to_list.items():
            try:
                # # Conditionally handle keys that should not add a new dimension (e.g., 'agents_in')
                # if key == 'obj_trajs' or key == 'obj_trajs_mask':
                #     # Concatenate along the desired axis if no new batch dimension is needed
                #     input_dict[key] = torch.cat(
                #         [torch.from_numpy(arr) for arr in val_list], dim=0)
                # else:
                #     # For other data, stack normally
                #     # input_dict[key] = torch.from_numpy(
                #     #     np.stack(val_list, axis=0))
                input_dict[key] = torch.cat(
                    [torch.from_numpy(arr) for arr in val_list], dim=0)
            except Exception as e:
                # Fallback to list if stacking fails
                input_dict[key] = val_list

        return {
            'batch_size': batch_size,
            'input_dict': input_dict,
            'batch_sample_count': batch_size
        }


class PlanTDataset(Dataset):
    """
    Create a dataset similar to the PlanT dataset structure.
    """

    def __init__(self, config: Dict[str, Any],
                 is_validation: bool = False,
                 include_ego: bool = False,
                 num_waypoints: int = 4) -> None:
        super().__init__()
        self.is_validation: bool = is_validation
        if is_validation:
            self.data_path: str = config['val_data_path']
        else:
            self.data_path: str = config['train_data_path']

        self.config: Dict[str, Any] = config
        self.num_samples: int = config['num_samples']
        self.include_ego: bool = include_ego
        self.num_waypoints: int = num_waypoints
        self.data = []  # Store processed data here
        self.num_workers: int = 8
        self.load_data()

    def load_data(self) -> None:
        """
        Load the dataset from the specified path in parallel.
        """
        if self.is_validation:
            print("Loading validation data...")
        else:
            print("Loading training data...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Path {self.data_path} does not exist")

        # Get all JSON files
        json_files: List[str] = glob.glob(
            os.path.join(self.data_path, "*.json")
        )

        if self.num_samples is not None:
            json_files = json_files[:self.num_samples]

        # Process files in parallel with specified number of workers
        # Use self.num_workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_file, j_file)
                       for j_file in json_files]
            for future in as_completed(futures):
                processed_samples = future.result()
                # Extend the main data list with processed samples
                self.data.extend(processed_samples)

    def process_file(self, j_file: str) -> List[Dict[str, Any]]:
        """
        Load and process data from a single JSON file.
        Returns a list of processed samples.
        """
        processed_samples = []
        with open(j_file) as f:
            data = json.load(f)
            for i in range(len(data) - self.num_waypoints):
                processed_sample = self.process_data(i, data, j_file)
                processed_samples.append(processed_sample)
        return processed_samples

    def process_data(self, idx: int, data: List, filename: str) -> Dict[str, Any]:
        """
        Process data for a single sample.
        """
        sample: Dict[str, Any] = {
            'scene_id': None,
            'input': [],
            'output': [],
            'waypoints': [],
            'bias_position': None,
            'filename': filename,
            'idx': idx
        }

        current_data: Dict[str, Any] = data[idx]

        if 'ego' in current_data:
            bias_position = current_data['ego']
            sample['bias_position'] = bias_position

        if 'vehicles' in current_data:
            for vehicle in current_data['vehicles']:
                vehicle_attributes: List = []
                vehicle_state_norm = np.array(
                    vehicle) - np.array(bias_position)
                vehicle_state_norm = vehicle_state_norm.tolist()
                vehicle_state_norm[3:] = vehicle[3:]
                # keep everything else the same
                vehicle_attributes.append(
                    ObjectTypes.PURSUERS.value)  # Example object type
                vehicle_attributes.extend(vehicle_state_norm)
                sample['input'].append(vehicle_attributes)

        # Add waypoints from the following frames
        next_waypoints = idx + self.num_waypoints
        for i in range(idx, next_waypoints):
            next_data = data[i]
            if 'ego' in next_data:
                waypoints = next_data['ego']
                waypoints_norm = np.array(waypoints) - np.array(bias_position)
                # convert back to list
                waypoints_norm = waypoints_norm.tolist()
                sample['waypoints'].append(waypoints_norm[0:2])

        return sample

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def collate_fn(self, data_batch):
        input_batch, output_batch = [], []
        bias_batch = []

        for element_id, sample in enumerate(data_batch):
            input_item = torch.tensor(sample["input"], dtype=torch.float32)
            output_item = torch.tensor(sample["output"])

            input_indices = torch.tensor(
                [element_id] * len(input_item)).unsqueeze(1)
            output_indices = torch.tensor(
                [element_id] * len(output_item)).unsqueeze(1)

            input_batch.append(torch.cat([input_indices, input_item], dim=1))
            output_batch.append(
                torch.cat([output_indices, output_item], dim=1))
        waypoints_batch = torch.tensor(
            [sample["waypoints"] for sample in data_batch])

        bias_batch = torch.tensor([sample["bias_position"]
                                   for sample in data_batch])
        # input_batch = torch.stack(input_batch)
        output_batch = torch.stack(output_batch)

        return {
            'input': input_batch,
            'output': output_batch,
            'waypoints': waypoints_batch,
            'bias_position': bias_batch
        }
