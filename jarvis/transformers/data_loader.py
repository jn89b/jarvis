import json
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter
# Assuming this is your transformer
from jarvis.transformers.test import CarTransformer


def open_json_file(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data


class CarTrajectoryDataset(Dataset):
    def __init__(self, data_files: List[dict]) -> None:
        self.data = data_files  # Each file contains a sequence of time steps

    def __len__(self) -> int:
        return len(self.data)  # Number of sequences

    def __getitem__(self, idx: int) -> Tuple[int, List[dict]]:
        # Return the index along with the entire sequence (multiple time steps)
        return idx, self.data[idx]


def collate_fn(batch: List[Tuple[int, List[dict]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    idx, sequence = batch[0]  # Since batch size is 1, just take the first item
    ego_seq = torch.tensor([step['ego']
                           for step in sequence], dtype=torch.float32)
    vehicles_seq = torch.tensor([step['vehicles']
                                for step in sequence], dtype=torch.float32)
    waypoints = torch.tensor([step['future_waypoints']
                             for step in sequence], dtype=torch.float32)

    # Pad sequences to make them consistent
    # Add batch dimension (1, seq_len, features)
    padded_ego = ego_seq.unsqueeze(0)
    # (1, seq_len, num_vehicles, features)
    padded_vehicles = vehicles_seq.unsqueeze(0)
    padded_waypoints = waypoints.unsqueeze(0)  # (1, seq_len, features)

    return padded_ego, padded_vehicles, padded_waypoints, idx
