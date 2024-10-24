from typing import Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel


def compute_distances(vehicles_data: Tensor) -> Tensor:
    """
    Computes the Euclidean distance between the ego vehicle and each pursuer.

    Args:
        ego_position (Tensor): Position of the ego vehicle with shape [batch_size, 2].
        vehicles_data (Tensor): Vehicles' data with shape [batch_size, num_vehicles, num_attributes].

    Returns:
        Tensor: Distance between the ego vehicle and each pursuer, shape [batch_size, num_vehicles].
    """
    distances = torch.norm(
        vehicles_data[:, :, :2], dim=-1)
    return distances


def adjust_attention_based_on_distance(attn_logits: Tensor, distances: Tensor, temperature: float = 1e-5) -> Tensor:
    """
    Adjusts attention logits by scaling them based on inverse distance to the ego vehicle.

    Args:
        attn_logits (Tensor): Raw attention logits from the transformer, shape [batch_size, num_heads, num_vehicles].
        distances (Tensor): Distances between the ego and each vehicle, shape [batch_size, num_vehicles].
        temperature (float): Temperature scaling to control the sharpness of attention distribution.

    Returns:
        Tensor: Adjusted attention weights, shape [batch_size, num_heads, num_vehicles].
    """
    # Compute inverse of the distances
    inverse_distances = 1 / (distances**2 + 1e-6)  # Prevent division by zero
    # inverse_distances = torch.exp(-distances)

    # Expand dimensions of inverse distances to match attention logits
    # Shape of inverse_distances: [batch_size, num_vehicles] --> [batch_size, 1, num_vehicles]
    inverse_distances = inverse_distances.unsqueeze(1)

    # Multiply inverse distances with the attention logits
    scaled_attn_logits = attn_logits * inverse_distances

    # Apply softmax with temperature scaling
    attention_weights = F.softmax(scaled_attn_logits/temperature, dim=-1)

    return attention_weights


class EvaderTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Define number of object types: only vehicles
        self.object_types: int = 1  # vehicles only (no ego)
        self.num_attributes: int = 4  # x, y, yaw, speed

        # Load pretrained model (checkpoint as per PlanT paper)
        hf_checkpoint: str = 'prajjwal1/bert-medium'
        self.model: nn.Module = AutoModel.from_pretrained(hf_checkpoint)
        n_embd: int = self.model.config.hidden_size  # Hidden size from the BERT model

        # Token embedding for vehicle data
        self.tok_emb: nn.Linear = nn.Linear(self.num_attributes, n_embd)

        # Object type embedding for vehicles
        self.obj_token: nn.Parameter = nn.Parameter(
            torch.randn(1, self.num_attributes))
        self.obj_emb: nn.Linear = nn.Linear(self.num_attributes, n_embd)
        self.drop: nn.Dropout = nn.Dropout(0.1)

        # Waypoint prediction head
        self.wp_head: nn.Linear = nn.Linear(n_embd, 64)
        self.wp_decoder: nn.GRUCell = nn.GRUCell(input_size=4, hidden_size=64)
        self.wp_relu: nn.ReLU = nn.ReLU()
        self.wp_output: nn.Linear = nn.Linear(64, 2)

    def forward(self, vehicles_data: Tensor, target_point: Tensor) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        Forward pass for predicting waypoints and returning attention maps.

        Args:
            vehicles_data (Tensor): Vehicles' data with shape [batch_size, num_vehicles, num_attributes].
            target_point (Tensor): Target future waypoint with shape [batch_size, 2].

        Returns:
            Tuple[Tensor, Tuple[Tensor]]: Predicted waypoints, attention maps.
        """
        B, O, A = vehicles_data.shape

        # Embed input tokens (vehicle data)
        input_batch_flat: Tensor = rearrange(
            vehicles_data, "b objects attributes -> (b objects) attributes")
        embedding: Tensor = self.tok_emb(input_batch_flat)
        embedding: Tensor = rearrange(
            embedding, "(b o) features -> b o features", b=B, o=O)

        # Add object type embedding for vehicles
        vehicle_embedding: Tensor = embedding + self.obj_emb(self.obj_token)

        # Apply dropout
        x: Tensor = self.drop(vehicle_embedding)

        # Pass through the transformer model
        output: Tuple[Tensor, Tuple[Tensor]] = self.model(
            **{"inputs_embeds": x}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
        # # Stack the adjusted attention maps
        # attn_map = torch.stack(attn_map, dim=0)

        distances = compute_distances(vehicles_data)

        # # Adjust attention based on distances
        # adjusted_attn_weights = []
        # for layer in attn_map:
        #     adjusted_layer_attn = adjust_attention_based_on_distance(
        #         layer[:, :, 0, 1:], distances)
        #     adjusted_attn_weights.append(adjusted_layer_attn)

        # # Stack the adjusted attention maps
        # attn_map = torch.stack(attn_map, dim=0)

        # Waypoint prediction
        z: Tensor = self.wp_head(x[:, 0, :])

        # Initialize prediction for the first waypoint
        output_wp: List[Tensor] = []
        pred_wp: Tensor = torch.zeros(
            (z.shape[0], 2), dtype=z.dtype).to(z.device)

        # Autoregressively generate waypoints
        for _ in range(4):  # Predict up to 4 waypoints
            x_in: Tensor = torch.cat(
                [pred_wp, target_point.unsqueeze(0)], dim=1)
            z: Tensor = self.wp_decoder(x_in, z)
            dx: Tensor = self.wp_output(z)
            pred_wp = pred_wp + dx
            output_wp.append(pred_wp)

        # Stack waypoints into shape [batch_size, num_waypoints, 2]
        pred_wp = torch.stack(output_wp, dim=1)
        return pred_wp, attn_map


def extract_avg_attention_with_distance(attn_weights: Tuple[Tensor], distances: Tensor) -> Tensor:
    """
    Extracts average attention from the attention maps and adjusts them based on the distance between pursuers and ego.

    Args:
        attn_weights (Tuple[Tensor]): Attention weights from the transformer, shape [num_layers, batch_size, num_heads, seq_length, seq_length].
        distances (Tensor): Distances between the ego and each pursuer, shape [batch_size, num_vehicles].

    Returns:
        Tensor: Adjusted average attention across layers, shape [batch_size, num_vehicles].
    """
    avg_attention_per_layer = []
    for layer_attention in attn_weights:
        # Extract attention to all vehicles
        # Assuming token 0 is ego and rest are vehicles
        ego_attention_all_tokens = layer_attention[:, :, 0, 1:]
        avg_attention_all_tokens = ego_attention_all_tokens.mean(
            dim=1)  # Average across heads
        adjusted_attention = adjust_attention_based_on_distance(
            avg_attention_all_tokens, distances)
        avg_attention_per_layer.append(adjusted_attention)

    # Stack and average across layers
    avg_attention_stacked = torch.stack(avg_attention_per_layer, dim=0)
    avg_attention = avg_attention_stacked.mean(dim=0)

    return avg_attention
