from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim

from torch import optim
from typing import Optional, Any, Dict
from jarvis.transformers.evadeformer_utils import (
    PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider)


from typing import Callable


def init(
    module: nn.Module,
    weight_init: Callable[[torch.Tensor, float], None],
    bias_init: Callable[[torch.Tensor], None],
    gain: float = 1.0
) -> nn.Module:
    '''
    This function provides weight and bias initializations for linear layers.

    Args:
        module (nn.Module): The module to initialize.
        weight_init (Callable): A function to initialize the weight, taking tensor and gain as inputs.
        bias_init (Callable): A function to initialize the bias, taking a tensor as input.
        gain (float): The gain to apply during weight initialization (default: 1).

    Returns:
        nn.Module: The initialized module.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class EvadeFormer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        def init_(m): return init(m, nn.init.xavier_normal_,
                                  lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr: int = config['num_map_feature']
        self.k_attr: int = config['num_agent_feature']
        self.d_k: int = config['hidden_size']
        self._M: int = config['max_num_agents']
        self.c: int = config['num_modes']
        self.T: int = config['future_len']
        self.num_queries_enc: int = config['num_queries_enc']
        self.num_queries_dec: int = config['num_queries_dec']

        # Define network layers
        self.road_pts_lin = nn.Sequential(
            init_(nn.Linear(self.map_attr, self.d_k)))
        self.agents_dynamic_encoder = nn.Sequential(
            init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(
            self.num_queries_enc, self.d_k)
        output_query_provider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'], num_query_channels=self.d_k, init_scale=0.1)
        self.perceiver_decoder = PerceiverDecoder(
            output_query_provider, self.d_k)
        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))
        self.output_model = nn.Sequential(
            init_(nn.Linear(self.d_k, 5 * self.T)))
        self.selu = nn.SELU(inplace=True)
        self.criterion = Criterion(config)

        # Positional Embeddings
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, self._M + 1, self.d_k)), requires_grad=True)
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, config['past_len'], 1, self.d_k)), requires_grad=True)

    def forward(self, batch: Dict):
        # Ensure tensors are on the model's device
        model_input = {}
        # Ensure tensors are on the model's device
        device = next(self.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)
                 else v for k, v in batch.items()}

        inputs = batch['input_dict']
        for k, v in inputs.items():
            model_input[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        track_index_to_predict = inputs['track_index_to_predict'].to(device)
        # flatten the array to a tensor
        track_index_to_predict = track_index_to_predict.view(-1)
        agents_in, agents_mask = inputs['obj_trajs'], inputs['obj_trajs_mask']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(
            1, 1,
            *agents_in.shape[
                -2:])).squeeze(1)

        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(
            -1, 1, 1).repeat(1, 1,
                             agents_mask.shape[
                                 -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        output = self._forward(model_input)
        # TODO: Remove this shit
        inputs['center_gt_trajs'][0, :, :2]
        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(output, ground_truth,
                              inputs['center_gt_final_valid_idx'])
        # loss = waypoint_mae_loss(output['predicted_trajectory'], ground_truth)
        output['dataset_name'] = 'test'  # inputs['dataset_name']
        output['predicted_probability'] = F.softmax(
            output['predicted_probability'], dim=-1)

        if not torch.isfinite(loss).all():
            print("Warning: Loss contains NaN or Inf values.")

        return output, loss

    def _forward(self, inputs):
        ego_in, agents_in = inputs['ego_in'], inputs['agents_in']
        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1

        agents_emb = self.selu(self.agents_dynamic_encoder(
            torch.cat((ego_in.unsqueeze(2), agents_in), dim=2)))
        agents_emb = (
            agents_emb + self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        mixed_input_features = torch.concat([agents_emb], dim=1)

        context = self.perceiver_encoder(mixed_input_features)
        out_seq = self.perceiver_decoder(context)
        out_dists = self.output_model(
            out_seq[:, :self.c]).reshape(B, self.c, self.T, -1)
        mode_probs = self.prob_predictor(
            out_seq[:, :self.c]).reshape(B, self.c)

        output = {
            'predicted_probability': F.softmax(mode_probs, dim=-1),
            'predicted_trajectory': out_dists,
            'scene_emb': out_seq[:, :self.num_queries_dec].reshape(B, -1)
        }
        return output

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        output, loss = self(batch)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        output, loss = self(batch)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.config['learning_rate'], eps=0.0001)

        # Calculate total_steps based on DataLoader length and total epochs
        # Assuming 'dataloader' is your train DataLoader
        # Define OneCycleLR with total_steps directly
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0002,
            epochs=100,
            steps_per_epoch=100,
            # total_steps=total_steps,
            pct_start=0.02,
            div_factor=100.0,
            final_div_factor=10
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class Criterion(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(Criterion, self).__init__()
        self.config: Dict[str, Any] = config

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        gt: torch.Tensor,
        center_gt_final_valid_idx: torch.Tensor,
        use_mae: bool = False
    ) -> torch.Tensor:

        return self.nll_loss_gmm_direct(
            out['predicted_probability'],
            out['predicted_trajectory'],
            gt,
            center_gt_final_valid_idx
        )

    def nll_loss_gmm_direct(
        self,
        pred_scores: torch.Tensor,
        pred_trajs: torch.Tensor,
        gt_trajs: torch.Tensor,
        center_gt_final_valid_idx: torch.Tensor,
        pre_nearest_mode_idxs: Optional[torch.Tensor] = None,
        timestamp_loss_weight: Optional[torch.Tensor] = None,
        use_square_gmm: bool = False,
        log_std_range: Tuple[float, float] = (-1.609, 5.0),
        rho_limit: float = 0.5
    ) -> torch.Tensor:
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_trajs.shape[0]
        gt_valid_mask = gt_trajs[..., -1]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] -
                        gt_trajs[:, None, :, :2]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
            nearest_mode_idxs = distance.argmin(dim=-1)

        nearest_mode_bs_idxs = torch.arange(
            batch_size).type_as(nearest_mode_idxs)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]
        res_trajs = gt_trajs[..., 0:2] - nearest_trajs[:, :, 0:2]

        # Extract and print ground truth and predicted trajectories as 2xN
        # for i in range(batch_size):
        #     gt_trajectory = gt_trajs[i, :, :2].T  # Shape (2, T)
        #     pred_trajectory = nearest_trajs[i, :, :2].T  # Shape (2, T)

        #     print(f"Ground Truth Trajectory (Sample {i}):\n", gt_trajectory)
        # print(f"Predicted Trajectory (Sample {i}):\n", pred_trajectory)

        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(
                nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(
                nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(
                nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)
            std2 = torch.exp(log_std2)
            rho = torch.clip(
                nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        reg_gmm_log_coefficient = log_std1 + \
            log_std2 + 0.5 * torch.log(1 - rho ** 2)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
            (dx ** 2) / (std1 ** 2) + (dy ** 2) /
            (std2 ** 2) - 2 * rho * dx * dy / (std1 * std2)
        )
        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp)
                    * gt_valid_mask).sum(dim=-1)
        loss_cls = F.cross_entropy(
            input=pred_scores, target=nearest_mode_idxs, reduction='none')

        # if torch.isnan(dx).any() or torch.isnan(dy).any():
        #     print("NaN in dx or dy values")
        # if torch.isnan(log_std1).any() or torch.isnan(log_std2).any():
        #     print("NaN in log_std values")
        # if torch.isnan(rho).any():
        #     print("NaN in rho values")

        # return (reg_loss + loss_cls).mean()
        loss = torch.abs(res_trajs).mean()

        return loss


def waypoint_mae_loss(pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error between predicted and ground truth waypoints.

    Args:
        pred_trajs (torch.Tensor): Predicted trajectories, shape (batch_size, num_waypoints, 2).
        gt_trajs (torch.Tensor): Ground truth trajectories, shape (batch_size, num_waypoints, 2).

    Returns:
        torch.Tensor: The MAE loss for the trajectories.
    """
    # Ensure the input shapes are compatible
    print("pred_trajs.shape: ", pred_trajs.shape)
    print("gt_trajs.shape: ", gt_trajs.shape)
    assert pred_trajs.shape == gt_trajs.shape, "Shape mismatch between predicted and ground truth trajectories."

    # Calculate Mean Absolute Error
    loss = torch.abs(pred_trajs - gt_trajs).mean()

    return loss
