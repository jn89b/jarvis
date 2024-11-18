from transformers import AutoModel
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CarTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Define number of object types: vehicles, ego, route
        self.object_types = 2  # + 1  # vehicles, route +1 for padding and waypoint embedding
        self.num_attributes = 4  # x, y, yaw, speed

        # Load pretrained model manually (checkpoint as per PlanT paper)
        hf_checkpoint = 'prajjwal1/bert-medium'  # Manually set the checkpoint
        self.model = AutoModel.from_pretrained(hf_checkpoint)
        n_embd = self.model.config.hidden_size  # Hidden size from the BERT model

        # CLS and EOS token embeddings for sequence
        self.cls_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))
        self.eos_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))

        # Token embedding for car/vehicle data
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)

        # Object type embedding (for vehicles and route)
        self.obj_token = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.num_attributes)) for _ in range(self.object_types)
        ])
        self.obj_emb = nn.ModuleList([
            nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)
        ])
        self.drop = nn.Dropout(0.1)  # Dropout with a fixed rate

        # Waypoint prediction head
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=64)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(64, 2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ego_data, vehicles_data, target_point):
        """
        ego_data: Ego vehicle data [batch_size, 1, num_attributes]
        vehicles_data: List of vehicle data [batch_size, num_vehicles, num_attributes]
        """

        # Ensure correct dimensionality for ego_data (batch_size, 1, num_attributes)
        ego_data = ego_data.unsqueeze(1) if ego_data.dim() == 2 else ego_data

        # Concatenate ego and vehicles data into a single input batch (sequence)
        # Resulting shape: [batch_size, 1 + num_vehicles, num_attributes]
        input_batch = torch.cat([ego_data, vehicles_data], dim=1)
        B, O, A = input_batch.shape  # B=batch_size, O=1 + num_vehicles, A=num_attributes

        # Embed input tokens
        input_batch_flat = rearrange(
            input_batch, "b objects attributes -> (b objects) attributes")
        embedding = self.tok_emb(input_batch_flat)
        embedding = rearrange(
            embedding, "(b o) features -> b o features", b=B, o=O)

        # Split embeddings for ego and vehicles
        # Ego embedding [batch_size, 1, features]
        ego_embedding = embedding[:, :1, :]  # Get the first token (ego)
        # Vehicle embeddings [batch_size, num_vehicles, features]
        # Get the rest of the tokens (vehicles)
        vehicle_embedding = embedding[:, 1:, :]

        # Add object type embeddings: ego gets its own embedding, and vehicles get their embedding
        ego_embedding = ego_embedding + \
            self.obj_emb[0](self.obj_token[0])  # Ego embedding adjustment
        vehicle_embedding = vehicle_embedding + \
            self.obj_emb[1](self.obj_token[1])  # Vehicles embedding adjustment

        # Concatenate back after applying the object-specific embeddings
        combined_embedding = torch.cat(
            [ego_embedding, vehicle_embedding], dim=1)

        # Apply dropout
        x = self.drop(combined_embedding)

        # Pass through the transformer model
        # output = self.model(inputs_embeds=x, output_attentions=True)
        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(
            **{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        # Waypoint prediction
        z = self.wp_head(x[:, 0, :])  # Get the first token (CLS-like)

        # Initialize prediction for the first waypoint
        output_wp = []
        # pred_wp = torch.zeros((z.shape[0], 2), dtype=z.dtype).to(z.device)
        # pred_wp

        x = torch.zeros((z.shape[0], 2), dtype=z.dtype).to(z.device)
        x = x.type_as(z)
        # Autoregressively generate waypoints
        for _ in range(4):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = torch.cat([x, target_point.unsqueeze(0)], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x = x + dx
            output_wp.append(x)

        # Stack waypoints into shape [batch_size, num_waypoints, 2]
        pred_wp = torch.stack(output_wp, dim=1)
        return pred_wp, attn_map

    def pad_sequence_batch(self, x_batched):
        x_batch_ids = x_batched[:, 0]
        x_tokens = x_batched[:, 1:]
        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []

        for batch_id in range(B):
            x_batch_id_mask = x_batch_ids == batch_id
            x_tokens_batch = x_tokens[x_batch_id_mask]
            x_seq = torch.cat(
                [self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)
            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]

        return input_batch
