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

    def __getitem__(self, idx: int) -> List[dict]:
        # Return the entire sequence (multiple time steps)
        return self.data[idx]

# Collate function for padding sequences


def collate_fn(batch: List[List[dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_seq = [torch.tensor([step['ego'] for step in b],
                            dtype=torch.float32) for b in batch]
    vehicles_seq = [torch.tensor(
        [step['vehicles'] for step in b], dtype=torch.float32) for b in batch]
    waypoints = [torch.tensor([step['future_waypoints']
                              for step in b], dtype=torch.float32) for b in batch]

    # Pad sequences
    padded_ego = pad_sequence(ego_seq, batch_first=True)
    padded_vehicles = pad_sequence(vehicles_seq, batch_first=True)
    padded_waypoints = pad_sequence(waypoints, batch_first=True)

    return padded_ego, padded_vehicles, padded_waypoints
