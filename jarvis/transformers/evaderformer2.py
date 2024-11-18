"""
From the paper:
    For PlanT, we extract the
    relevance score by adding the attention weights of all layers and
    heads for the [CLS] token.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange
from typing import Optional, Dict, Any, Tuple, Callable
from torch import optim


from transformers import (
    AutoConfig,
    AutoModel,
    PerceiverConfig,
    PerceiverModel,
)


class EvaderFormer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config: Dict[str, Any] = config
        self.object_types: int = 1  # vehicles + padding and wp embedding
        # self.num_attributes: int = 6  # x, y, psi, v, vx, vy
        self.num_attributes: int = 7  # for uav x,y,z, roll, pitch, yaw, v
        self.wp_size: int = 3
        self.num_wps_predict: int = 4
        hf_checkpoint: str = 'prajjwal1/bert-medium'
        self.model: nn.Module = AutoModel.from_pretrained(hf_checkpoint)
        n_embd: int = self.model.config.hidden_size

        self.cls_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
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

        self.criterion = nn.L1Loss()  # L1 loss for waypoint prediction

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

    def forward(self, batch, target=None):
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
        ]
        # stack becomes [1,1,n_pursuers+cls, d_head]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)
        # embedding dropout
        x = self.drop(embedding)
        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(
            **{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
       # get waypoint predictions
        z = self.wp_head(x[:, 0, :])
        # add traffic ligth flag
        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], self.wp_size), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        last_waypoint = waypoints[:, -1, :]
        for _ in range(self.num_wps_predict):
            x_in = torch.cat([x, last_waypoint], dim=1)

            z = self.wp_decoder(x_in[:, self.wp_size-1:], z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        logits = None
        return logits, pred_wp, attn_map

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
