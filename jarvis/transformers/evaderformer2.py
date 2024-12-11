"""
From the paper:
    For PlanT, we extract the
    relevance score by adding the attention weights of all layers and
    heads for the [CLS] token.
https://medium.com/@geetkal67/attention-networks-a-simple-way-to-understand-self-attention-f5fb363c736d

"""
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torch.nn.utils.rnn import pad_sequence
from torch.func import vmap
import torch.multiprocessing as mp
import time
from einops import rearrange
from typing import Optional, Dict, Any, Tuple, Callable
from torch import optim
from functools import partial

from transformers import (
    AutoConfig,
    AutoModel,
    PerceiverConfig,
    PerceiverModel,
)


class EvaderFormer(pl.LightningModule):
    """
    Include a decoder network to om
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config: Dict[str, Any] = config
        self.object_types: int = 1  # vehicles + padding and wp embedding
        # self.num_attributes: int = 6  # x, y, psi, v, vx, vy
        self.num_attributes: int = 7  # for uav x,y,z, roll, pitch, yaw, v
        self.wp_size: int = 3
        self.num_wps_predict: int = 4
        hf_checkpoint: str = 'prajjwal1/bert-medium'
        self.model: nn.Module = AutoModel.from_pretrained(hf_checkpoint,
                                                          output_hidden_states=True)
        n_embd: int = self.model.config.hidden_size

        self.cls_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # this is used for unique objects
        self.obj_token = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.num_attributes)) for _ in range(self.object_types)
        ])
        self.obj_emb = nn.ModuleList([
            nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)
        ])

        self.drop = nn.Dropout(0.1)

        # Waypoint decoder
        self.wp_head = nn.Linear(n_embd, 65)
        self.wp_decoder = nn.GRUCell(
            input_size=self.num_wps_predict, hidden_size=65)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(65, self.wp_size)
        self.criterion = nn.L1Loss()  # L1 loss for waypoint prediction#

        # Let's see if we can add a gradcam to visualize the weights of our neight
        # call a hook to the last layer of the model
        # self.reduction_layer = nn.Linear(n_embd, self.num_attributes)
        # self.hook_handle = self.model.encoder.layer[-1].register_forward_hook(
        #     self.my_hook)

    # def my_hook(self, module, input, output):
    #     # print(""module)
    #     print("input", input[0].shape, input[1].shape)
    #     print("output", output[0].shape, output[1].shape, output[2].shape)
    #     print("hook called")

    def pad_sequence_batch(self, x_batched):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # split input into components
        x_batch_ids = x_batched[:, 0]

        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []
        for batch_id in range(B):
            # get the batch of elements
            x_batch_id_mask = x_batch_ids == batch_id

            # get the batch of types
            x_tokens_batch = x_tokens[x_batch_id_mask]

            x_seq = torch.cat(
                [self.cls_emb, x_tokens_batch], dim=0)

            input_batch.append(x_seq)

        # pad sequence is used to pad the sequence to the
        # longest sequence in the batch
        # so for example if we have a batch of 3 sequences
        # with lengths 3, 4, 5
        # the sequences will be padded to length 5
        # so the output will be of size 5 x 3

        # we then use swapaxes to get the batch first and then the sequence
        # so for example if it was 3 x 1 x 5
        # it will be 1 x 3 x 5
        # I might use einops to do this instead
        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]
        return input_batch

    def forward(self, batch, target=None,
                return_interpretability: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = batch['input']
        waypoints = batch['waypoints']
        x_batched = torch.cat(idx, dim=0)
        input_batch = self.pad_sequence_batch(x_batched)
        # we get the object types from this line
        input_batch_type = input_batch[:, :, 0]
        input_batch_data = input_batch[:, :, 1:]

        # create masks by object type
        car_mask = (input_batch_type == 2).unsqueeze(-1)
        # route_mask = (input_batch_type == 2).unsqueeze(-1)
        masks = [car_mask]

        # get size of input
        # batch size, number of objects, number of attributes
        (B, O, A) = (input_batch_data.shape)
        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(
            embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)
        ]  # list of a tensors of size 1 x features
        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]  # list of tensors of size B x O x features
        # stack becomes [1,1,n_pursuers+cls, d_head]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)
        # embedding dropout
        x = self.drop(embedding)
        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(
            **{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
       # get waypoint predictions
        z = self.wp_head(x[:, 0, :])  # [1, 65]
        # add traffic ligth flag
        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], self.wp_size), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        last_waypoint = waypoints[:, -1, :]
        for _ in range(self.num_wps_predict):
            x_in = torch.cat([x, last_waypoint], dim=1)  # [1,6]

            z = self.wp_decoder(x_in[:, self.wp_size-1:], z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        logits = None
        return logits, pred_wp, attn_map

    def grad_rollout(self, attentions, gradients, discard_ratio: float = 0.9):
        result = torch.eye(attentions[0].size(-1)).to(attentions.device)
        with torch.no_grad():
            for attention, grad in zip(attentions, gradients):
                weights = grad
                attention_heads_fused = (attention*weights).mean(axis=1)
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(
                    attention_heads_fused.size(0), -1)
                _, indices = flat.topk(
                    int(flat.size(-1)*discard_ratio), -1, False)
                # indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        return result

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        waypoints = batch['waypoints']
        _, pred_wp, _ = self(batch)
        # Compute L1 loss between predicted and true waypoints
        loss = self.criterion(pred_wp, waypoints)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        waypoints = batch['waypoints']
        _, pred_wp, _ = self(batch)

        # Compute L1 loss for validation
        loss = self.criterion(pred_wp, waypoints)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), eps=0.0001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0002,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.02,
            div_factor=100.0,
            final_div_factor=10
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PursuerAttentionNetwork(nn.Module):
    def __init__(self, n_attributes: int, d_model: int) -> None:
        """
        Initialize the Pursuer Attention Network.

        Args:
            n_attributes (int): Number of attributes (e.g., position, velocity, heading).
            d_model (int): Higher-dimensional latent space for attention.
        """
        super(PursuerAttentionNetwork, self).__init__()
        self.n_attributes = n_attributes
        self.d_model = d_model

        # Projection layers for Q, K, V
        self.query_proj = nn.Linear(n_attributes, d_model)
        self.key_proj = nn.Linear(n_attributes, d_model)
        self.value_proj = nn.Linear(n_attributes, d_model)

        # Back-projection to attribute space
        self.output_proj = nn.Linear(d_model, n_attributes)

        # Learnable attribute weights for scaling the final output
        self.attr_weights = nn.Parameter(torch.ones(n_attributes))

    def forward(self, pursuer_attributes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Pursuer Attention Network.

        Args:
            pursuer_attributes (torch.Tensor): Input tensor of shape
                                               (batch_size, n_pursuers, n_attributes).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, n_pursuers, n_attributes).
        """
        batch_size, n_pursuers, _ = pursuer_attributes.size()

        # Project attributes into higher-dimensional space
        # Shape: (batch_size, n_pursuers, d_model)
        Q = self.query_proj(pursuer_attributes)
        # Shape: (batch_size, n_pursuers, d_model)
        K = self.key_proj(pursuer_attributes)
        # Shape: (batch_size, n_pursuers, d_model)
        V = self.value_proj(pursuer_attributes)

        # Compute scaled dot-product attention
        # Shape: (batch_size, n_pursuers, n_pursuers)
        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        # Shape: (batch_size, n_pursuers, n_pursuers)
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Apply attention to values
        # Shape: (batch_size, n_pursuers, d_model)
        attention_output = torch.matmul(attention_probs, V)
        # Back-project to attribute space
        # Shape: (batch_size, n_pursuers, n_attributes)
        reduced_output = self.output_proj(attention_output)
        # Scale by learnable attribute weights
        # Shape: (batch_size, n_pursuers, n_attributes)
        scaled_attention = reduced_output * \
            self.attr_weights.unsqueeze(0).unsqueeze(0)
        # apply softmax to the attention weights
        # scaled_attention
        return scaled_attention


class HEvadrFormer(EvaderFormer):
    """
    Let's try to train an Ensemble Method
    https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Multimodal_Motion_Prediction_With_Stacked_Transformers_CVPR_2021_paper.pdf 
    - Using an ensemble of transformers to get an attention value for each pursuer
    - Each of these transformers will give me an attention value for 
        each pursuer based on the CLS token rows
    - I will then use this as the features for the final transformer  

    - Tokenize each of the attributes 
    - From that we will need to embed the tokens to a higher dimension
    - We will then use the transformers to get the attention values

    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)
        self.num_attributes: int = 5  # x, y, z, roll, pitch, yaw, v
        self.cls_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))

        self.num_pursuers: int = config.get('num_pursuers', 3)

        hf_checkpoint: str = 'prajjwal1/bert-small'

        # Initialize the ensemble of transformers
        self.models = nn.ModuleList([AutoModel.from_pretrained(
            hf_checkpoint) for _ in range(self.num_pursuers)])
        # self.model: nn.Module = AutoModel.from_pretrained(hf_checkpoint)
        n_embd: int = self.model.config.hidden_size
        self.token_embds = nn.ModuleList(
            [nn.Linear(1, n_embd) for _ in range(self.num_pursuers)])
        self.cls_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.num_attributes + 1)) for _ in range(self.num_pursuers)])
        self.global_cls_emb = nn.ParameterList(
            [nn.Parameter(torch.randn(1, n_embd))
             for _ in range(self.num_pursuers)]
        )
        self.drops = nn.ModuleList([nn.Dropout(0.1)
                                   for _ in range(self.num_pursuers)])

    def forward_single_pursuer(self, pursuer_idx, input_batch_data, token_embd, model):
        """
        Forward pass for a single pursuer.
        """
        B, O, A = input_batch_data.shape

        # Tokenize attributes
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects attributes) 1"
        )
        embedding = token_embd(input_batch_data)
        embedding = rearrange(
            embedding, "(b o a) d -> b (o a) d", b=B, o=O, a=A
        )

        # Forward pass
        output = model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
        return x, attn_map

    def parallel_forward(self, pursuer_inputs):
        """
        Parallelize the forward pass for all pursuers.
        """
        with mp.Pool(processes=len(self.models)) as pool:
            results = pool.starmap(
                self.forward_single_pursuer,
                pursuer_inputs
            )
        return results

    def pad_individual_seq(self, x_batched, idx_layer: int):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # check if x_batched is 2D
        x_batch_ids = x_batched[:, 0]
        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []

        for batch_id in range(B):
            x_batch_id_mask = x_batch_ids == batch_id
            x_tokens_batch = x_tokens[x_batch_id_mask]
            x_seq = torch.cat(
                [self.cls_embs[idx_layer], x_tokens_batch], dim=0)
            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]
        return input_batch

    def forward(self, batch, target=None):
        """
        batch['input']: List[tensor] tensor is of shape [num_pursuers, num_attributes]
        batch['waypoints']: tensor of shape [batch_size, num_waypoints, (x,y,z)]
        """
        idx = batch['input']
        waypoints = batch['waypoints']
        x_batched = torch.cat(idx, dim=0)

        num_pursuers, num_attributes = x_batched.shape
        num_batches = num_pursuers // self.num_pursuers
        pursuer_attributes = []
        attn_maps = []
        # start_time = time.time()
        for i in range(num_pursuers):
            # x_batched[i] = x_batched[i].unsqueeze(0)  # [1, num_attributes]
            pursuer_idx = i % self.num_pursuers  # Ensure valid index for self.cls_embs
            squeezed_batch = x_batched[pursuer_idx].unsqueeze(0)
            input_batch = self.pad_individual_seq(
                squeezed_batch, pursuer_idx)  # becomes a [1, cls + current_pursuer, cls+num_attributes]
            pursuer_attributes.append(
                input_batch)

            input_batch_type = input_batch[:, 1:, 0]
            input_batch_data = input_batch[:, 1:, 2:]

            # Create masks by object type
            car_mask = (input_batch_type == 2).unsqueeze(-1)
            masks = [car_mask]

            # Size of input
            # [1, cls+pursuer, num_attributes]
            (B, O, A) = input_batch_data.shape
            # Tokenize attributes individually
            input_batch_data = rearrange(
                input_batch_data, "b objects attributes -> (b objects attributes) 1"
            )  # Flatten to treat each attribute as a token
            # Shape: (B * O * A, n_embd)
            embedding = self.token_embds[pursuer_idx](input_batch_data)
            # embedding = rearrange(
            #     embedding, "(b o a) d -> b (a) d", b=B, o=O, a=A
            # )  # Reshape back to batch-first format
            cls_token = self.global_cls_emb[pursuer_idx]
            embedding = torch.cat([cls_token, embedding], dim=0)
            x = self.drops[pursuer_idx](embedding)
            embedding = rearrange(embedding, 'rows dim -> 1 rows dim')

            # I want this to be [1,1,num_attributes+cls, d_head] right now its [1, num_attributes*num_cls, d_head]
            output = self.models[pursuer_idx](
                **{"inputs_embeds": embedding}, output_attentions=True)
            x, attn_map = output.last_hidden_state, output.attentions   \

            attn_maps.append(attn_map)

        # Waypoint prediction (same as before)
        x = x.repeat(num_batches, 1, 1)
        z = self.wp_head(x[:, 0, :])
        output_wp = list()
        x = torch.zeros(
            size=(z.shape[0], self.wp_size), dtype=z.dtype).type_as(z)
        last_waypoint = waypoints[:, -1, :]

        for _ in range(self.num_wps_predict):
            x_in = torch.cat([x, last_waypoint], dim=1)
            z = self.wp_decoder(x_in[:, self.wp_size - 1:], z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        logits = None
        return logits, pred_wp, attn_maps

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        waypoints = batch['waypoints']
        _, pred_wp, _ = self(batch)
        # Compute L1 loss between predicted and true waypoints
        loss = self.criterion(pred_wp, waypoints)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        waypoints = batch['waypoints']
        _, pred_wp, _ = self(batch)

        # Compute L1 loss for validation
        loss = self.criterion(pred_wp, waypoints)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), eps=0.0001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0002,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.02,
            div_factor=100.0,
            final_div_factor=10
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# # # Input: Batch of 2 pursuers, each with 7 attributes
# batch_size = 1
# n_pursuers = 4  # Number of pursuers
# n_attributes = 7  # Attributes: e.g., position, velocity, heading
# d_model = 8  # Dimensionality of attention mechanism

# # Generate test input: random attributes for each pursuer in each batch
# # Shape: (batch_size, n_pursuers, n_attributes)
# test_input = torch.rand(batch_size, n_pursuers, n_attributes)

# # Initialize the network
# model = PursuerAttentionNetwork(n_attributes=n_attributes, d_model=d_model)
# output = model(test_input)
# print("output", output.shape)

# hevader = HEvadrFormer(config={})
