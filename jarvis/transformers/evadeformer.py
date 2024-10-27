from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class EvadeFormer(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config: Dict[str, Any] = config

        def init_(m): return init(m, nn.init.xavier_normal_,
                                  lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.fisher_information: Optional[torch.Tensor] = None
        self.map_attr: int = config['num_map_feature']
        self.k_attr: int = config['num_agent_feature']
        self.d_k: int = config['hidden_size']
        self._M: int = config['max_num_agents']
        self.c: int = config['num_modes']
        self.T: int = config['future_len']
        self.L_enc: int = config['num_encoder_layers']
        self.dropout: float = config['dropout']
        self.num_heads: int = config['tx_num_heads']
        self.L_dec: int = config['num_decoder_layers']
        self.tx_hidden_size: int = config['tx_hidden_size']
        self.use_map_img: bool = config['use_map_image']
        self.use_map_lanes: bool = config['use_map_lanes']
        self.past_T: int = config['past_len']
        self.max_points_per_lane: int = config['max_points_per_lane']
        self.max_num_roads: int = config['max_num_roads']
        self.num_queries_enc: int = config['num_queries_enc']
        self.num_queries_dec: int = config['num_queries_dec']

        # Network Layers and Modules
        self.road_pts_lin: nn.Sequential = nn.Sequential(
            init_(nn.Linear(self.map_attr, self.d_k)))

        self.agents_dynamic_encoder: nn.Sequential = nn.Sequential(
            init_(nn.Linear(self.k_attr, self.d_k)))

        self.perceiver_encoder: PerceiverEncoder = PerceiverEncoder(
            self.num_queries_enc, self.d_k,
            num_cross_attention_qk_channels=self.d_k,
            num_cross_attention_v_channels=self.d_k,
            num_self_attention_qk_channels=self.d_k,
            num_self_attention_v_channels=self.d_k
        )

        output_query_provider: TrainableQueryProvider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'],
            num_query_channels=self.d_k,
            init_scale=0.1,
        )

        # Positional Embeddings
        self.agents_positional_embedding: nn.Parameter = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding: nn.Parameter = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        # Optional map positional embedding
        # self.map_positional_embedding: nn.Parameter = nn.parameter.Parameter(
        #     torch.zeros((1, self.max_points_per_lane * self.max_num_roads, self.d_k)), requires_grad=True
        # )

        self.perceiver_decoder: PerceiverDecoder = PerceiverDecoder(
            output_query_provider, self.d_k)

        self.prob_predictor: nn.Sequential = nn.Sequential(
            init_(nn.Linear(self.d_k, 1)))

        self.output_model: nn.Sequential = nn.Sequential(
            init_(nn.Linear(self.d_k, 5 * self.T)))

        self.selu: nn.SELU = nn.SELU(inplace=True)

        self.criterion: Criterion = Criterion(config)

        self.fisher_information: Optional[torch.Tensor] = None
        self.optimal_params: Optional[torch.Tensor] = None
        self.optimizer = optim.AdamW(
            self.parameters(), lr=config['learning_rate'], eps=0.0001)

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat(
            (torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''

        ego_in, agents_in = inputs['ego_in'], inputs['agents_in']

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(
            ego_in, agents_in)
        agents_tensor = torch.cat(
            (ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                                                    :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        mixed_input_features = torch.concat(
            [agents_emb], dim=1)
        mixed_input_masks = torch.concat(
            [opps_masks_agents.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(
            mixed_input_features, mixed_input_masks)

        # Wazformer-Ego Decoding

        out_seq = self.perceiver_decoder(context)

        out_dists = self.output_model(
            out_seq[:, :self.c]).reshape(B, self.c, self.T, -1)

        # Mode prediction

        mode_probs = self.prob_predictor(
            out_seq[:, :self.c]).reshape(B, self.c)

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['predicted_probability'] = mode_probs  # #[B, c]
        # [B, c, T, 5] to be able to parallelize code
        output['predicted_trajectory'] = out_dists
        output['scene_emb'] = out_seq[:, :self.num_queries_dec].reshape(B, -1)
        if len(np.argwhere(np.isnan(out_dists.detach().cpu().numpy()))) > 1:
            breakpoint()
        return output

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

        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(output, ground_truth,
                              inputs['center_gt_final_valid_idx'])
        output['dataset_name'] = 'test'  # inputs['dataset_name']
        output['predicted_probability'] = F.softmax(
            output['predicted_probability'], dim=-1)
        return output, loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.config['learning_rate'], eps=0.0001)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0002, steps_per_epoch=1, epochs=150,
                                                        pct_start=0.02, div_factor=100.0, final_div_factor=10)

        return [optimizer], [scheduler]


class Criterion(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(Criterion, self).__init__()
        self.config: Dict[str, Any] = config

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        gt: torch.Tensor,
        center_gt_final_valid_idx: torch.Tensor
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
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Args:
            pred_scores: Tensor of shape (batch_size, num_modes)
            pred_trajs: Tensor of shape (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs: Tensor of shape (batch_size, num_timestamps, 3)
            center_gt_final_valid_idx: Tensor of valid indices
            pre_nearest_mode_idxs: Optional tensor of mode indices
            timestamp_loss_weight: Optional tensor of loss weights per timestamp
            use_square_gmm: Boolean flag for square GMM usage
            log_std_range: Range for standard deviation clipping
            rho_limit: Limit for correlation coefficient

        Returns:
            torch.Tensor: The calculated GMM loss.
        """
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
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]
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

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

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

        return (reg_loss + loss_cls).mean()
