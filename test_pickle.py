import pickle as pkl
import numpy as np
from typing import Dict, Tuple, Any
from enum import Enum
# open data.pkl file in read mode


class StateIndex(Enum):
    X = 0
    Y = 1
    YAW = 2
    VEL = 3
    VX = 4
    VY = 5


with open('sample.pkl', 'rb') as file:
    sample = pkl.load(file)

with open('intermediate.pkl', 'rb') as file:
    inter_data= pkl.load(file)


### Testing what is going on in the process data function
info = inter_data
current_time_index:int = 21
track_infos: Dict[str, Any] = info['track_info']
obj_types = np.array(track_infos['object_type'])
# (num_objects, num_timestamp, 10)
obj_trajs_full: np.array = track_infos['trajs']
obj_trajs_past: np.array = obj_trajs_full[:, :current_time_index + 1]
obj_trajs_future: np.array = obj_trajs_full[:, current_time_index + 1:]

num_center_objects = 3

track_index_to_predict = np.array(
    info['tracks_to_predict']['track_index'])

obj_trajs_future: np.array = obj_trajs_full[:, current_time_index + 1:]
# let's see what happens when we try to get the gt_mask
obj_trajs_future = np.tile(
    obj_trajs_future[None, :, :, :], (num_center_objects, 1, 1, 1))

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


with open('output_data.pkl', 'rb') as file:
    output_data= pkl.load(file)


with open("center_gt_trajs.pkl", "rb") as file:
    gt_trajs = pkl.load(file)