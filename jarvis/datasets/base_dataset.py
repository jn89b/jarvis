import json
import ijson
import concurrent.futures
import glob
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any, Optional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter
from jarvis.datasets.common_utils import generate_mask
from collections import defaultdict

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

            for s in split:
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
            print(f"Loaded {len(sample_batches)} samples from {k}")
            for samples in sample_batches:
                samples: Dict[str, Any]

                output: Dict[str, Any] = self.preprocess_data(
                    data=samples, filename=k, idx=i)

                self.process_data(output)

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
            Tuple containing:
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

        Maybe better to return this as a dictionary... 
        """

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape

        # the data is already made to be relative to the center object or the ego vehicle
        # Expand obj_trajs_past to have an additional dimension for num_center_objects
        object_trajs = np.tile(obj_trajs_past, (num_center_objects, 1, 1, 1))
        # Now object_trajs will have shape (num_center_objects, num_objects, num_timestamps, box_dim)
        print("object_trajs shape:", object_trajs.shape)

        return

    def process_data(self, internal_format: Dict) -> None:
        """
        Process the loaded data
        """
        info: Dict[str, Any] = internal_format
        current_time_index: int = info['current_time_index']
        # timestamps: np.array = np.array(
        #     info['time'][:current_time_index + 1], dtype=np.float32)
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

        self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict_selected,
            sdc_track_index=0,
            timestamps=np.array(info['time']),
            obj_types=obj_types
        )

    def __len__(self):
        pass
        # return len(self.data_loaded)

    def __getitem__(self, idx: int):
        pass
        # if self.config['store_data_in_memory']:
        #     return self.data_loaded_memory[idx]
        # else:
        #     with open(self.data_loaded_keys[idx], 'rb') as f:
        #         return pickle.load(f)
