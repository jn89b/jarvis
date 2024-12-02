"""
Used to help deploy and apply transformer models to real time data
This is a test script to see how this works

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pkl
import yaml
import os

from typing import Dict, List, Tuple
from jarvis.transformers.evaderformer2 import EvaderFormer


class AgentHistory():
    def __init__(self) -> None:
        self.x: List[float] = []
        self.y: List[float] = []
        self.psi: List[float] = []
        self.attention_scores: List[float] = []


class Agent():
    def __init__(self,
                 x: float,
                 y: float,
                 z: float,
                 psi: float,
                 dt: float = 0.1) -> None:

        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.psi: float = psi
        self.dt: float = dt
        self.v: float = 0
        self.vx: float = 0
        self.vy: float = 0
        self.agent_history: AgentHistory = AgentHistory()
        self.update_history()

    def update_history(self) -> None:
        self.agent_history.x.append(self.x)
        self.agent_history.y.append(self.y)
        self.agent_history.psi.append(self.psi)

    def move(self, v: float, psi_dot: float) -> None:
        self.x += v * np.cos(self.psi) * self.dt
        self.y += v * np.sin(self.psi) * self.dt
        self.psi += psi_dot * self.dt
        self.v = v
        self.vx = v * np.cos(self.psi)
        self.vy = v * np.sin(self.psi)
        self.update_history()


class EvaderFormerUtils():
    """
    Class
    """
    # @classmethod

    def batch_data(self, ego: Agent, pursuers: List[float]) -> Dict[str, List[Tuple]]:
        """
        Load the batch data
        For the evaderformer it takes in a batch which is formatted as a Dict
        'input': List[Tuple]
        'waypoints': List[Tuple]

        The input key consists of each pursuer with the following format:
            - CLS token: 0
            - Object ID: for pursuer it is 2
            - relative x position wrt to ego vehicle
            - relative y position wrt to ego vehicle
            - relative psi angle wrt to ego vehicle
            - relative speed wrt to ego vehicle
            - relativce vx velocity wrt to ego vehicle
            - relative vy velocity wrt to ego vehicle

        waypoints consist of a tuple of the x and y coordinates for the ego
        for N number of waypoints
        """
        batch_data: Dict[str, List[Tuple]] = {
            'input': [],
            'waypoints': [],
            'bias_position': []
        }
        for p in pursuers:
            dx = ego.x - p.x
            dy = ego.y - p.y
            dz = ego.z - p.z
            psi = ego.psi - p.psi
            v = ego.v - p.v
            vx = ego.vx - p.vx
            vy = ego.vy - p.vy
            dx -= ego.x
            dy -= ego.y

            pursuer_state = [(0, 2, dx, dy, dz, psi, v, vx, vy)]
            batch_data['input'].append(pursuer_state)

        N_waypoints = 4
        waypoints = []
        for i in range(N_waypoints):
            coordinates = (5, 10, 50)
            waypoints.append(coordinates)

        # torn the batch_data into a torch tensor
        batch_data['input'] = [torch.tensor(
            batch_data['input'], dtype=torch.float32).squeeze()]

        batch_data['bias_position'] = torch.tensor(
            [ego.x, ego.y, ego.z,
             ego.psi, ego.v, ego.vx, ego.vy], dtype=torch.float32).squeeze()

        batch_data['waypoints'] = torch.tensor(
            [waypoints], dtype=torch.float32)

        batch_data['output'] = None

        return batch_data

    def compute_attention_scores(self,
                                 attention_map: Tuple[torch.tensor]) -> np.ndarray:
        """
        Compute the normalized attention scores from the cls token
        """
        # Shape: [batch_size, sequence_length]
        batch_size, num_tokens = attention_map[0].shape[0], attention_map[0].shape[-1]
        relevance_scores = torch.zeros(batch_size, num_tokens)

        # Sum attention weights across all layers and heads
        for layer_attention in attention_map:
            # Sum over heads to get the attention distribution of the [CLS] token across the sequence
            cls_attention = layer_attention[:, :, 0, :].sum(
                dim=1)  # Shape: [batch_size, sequence_length]
            relevance_scores += cls_attention  # Accumulate across layers

        # Average across the batch if you have multiple samples and want a single relevance score per token
        avg_relevance_scores = relevance_scores.mean(
            dim=0).detach().numpy()  # Shape: [sequence_length]

        # normalize the scores
        normalized_scores = avg_relevance_scores / avg_relevance_scores.sum()
        return normalized_scores


def pure_pursuit(ego: Agent, target: Agent) -> float:
    """
    Pure pursuit algorithm
    """
    dx: float = target.x - ego.x
    dy: float = target.y - ego.y
    los: float = np.arctan2(dy, dx)
    heading_cmd = los - ego.psi
    if heading_cmd > np.pi:
        heading_cmd = heading_cmd - 2 * np.pi
    elif heading_cmd < -np.pi:
        heading_cmd = heading_cmd + 2 * np.pi

    return heading_cmd


if __name__ == "__main__":

    # TODO: Need to refactor and modularize the code
    # I have a lot of boiler plate code that can be refactored
    data_config_path = "config/data_config.yaml"
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # Set up the dataset and dataloader
    # dataset = PlanTDataset(config=data_config, is_validation=True)
    # dataloader = DataLoader(dataset, batch_size=1,
    #                         shuffle=False, collate_fn=dataset.collate_fn)

    # Load the latest checkpoint
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

    data = pkl.load(open("batch.pkl", "rb"))
    test_input = data['input']
    waypoints = data['waypoints']
    evader_utiils = EvaderFormerUtils()

    ego = Agent(0, 0, 30, np.deg2rad(45))

    pursuer_1 = Agent(-150, -75, 50, 0)
    pursuer_2 = Agent(0, -200, 50, 0)
    pursuers = [pursuer_1, pursuer_2]
    N_steps: int = 250
    pursuer_indices = [1, 2]
    waypoints_history = []
    import time
    for i in range(N_steps):
        print("i", i)
        # Get the heading command for the pursuers
        heading_cmd_1 = pure_pursuit(ego=pursuer_1, target=ego)
        heading_cmd_2 = pure_pursuit(ego=pursuer_2, target=ego)
        batch_data: Dict[str, List[Tuple]
                         ] = evader_utiils.batch_data(ego, pursuers)
        # Move the pursuers
        start_time = time.time()
        _, predicted_waypoints, attn_map = model(batch_data)
        final_time = time.time() - start_time
        print("final time", final_time)
        predicted_waypoints = predicted_waypoints.detach().numpy().squeeze()
        norm_attention_scores = evader_utiils.compute_attention_scores(
            attn_map)
        pursuer_relevance_scores = norm_attention_scores[pursuer_indices]
        bias_position = batch_data['bias_position'].detach().numpy().squeeze()
        global_predicted_waypoints = predicted_waypoints + bias_position[:3]
        waypoints_history.extend([global_predicted_waypoints[-1]])
        ego.move(20, 0)
        pursuer_1.move(15, heading_cmd_1)
        pursuer_2.move(30, heading_cmd_2)

        for i, p in enumerate(pursuers):
            p.agent_history.attention_scores.append(
                pursuer_relevance_scores[i])

    fig, ax = plt.subplots()
    ax.plot(ego.agent_history.x, ego.agent_history.y, label="Evader")
    ax.plot(pursuer_1.agent_history.x,
            pursuer_1.agent_history.y, label="Pursuer 1", color='red')
    ax.plot(pursuer_2.agent_history.x,
            pursuer_2.agent_history.y, label="Pursuer 2", color='green')
    for p in pursuers:
        ax.plot(p.agent_history.x[0], p.agent_history.y[0], 'o', label="Start")
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(pursuer_1.agent_history.attention_scores,
            label="Pursuer 1", color='red')
    ax.plot(pursuer_2.agent_history.attention_scores,
            label="Pursuer 2", color='green')
    ax.legend()

    fig, ax = plt.subplots()
    # convert list to array
    waypoints_history = np.array(waypoints_history)
    ax.plot(waypoints_history[:, 0], waypoints_history[:, 1])

    plt.show()
