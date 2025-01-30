import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# from unitraj.models.base_model.base_model import BaseModel
from jarvis.transformers.wayformer.base_model import BaseModel

from jarvis.transformers.wayformer.wayformer_utils import \
    (PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider)


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PredictFormer(BaseModel):
    def __init__(self, config):
        super(PredictFormer, self).__init__(config)
        self.config = config

        def init_(m): return init(m, nn.init.xavier_normal_,
                                  lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.fisher_information = None
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.d_k = config['hidden_size']
        self._M = config['max_num_agents']  # num agents without the ego-agent
        self.c = config['num_modes']
        self.T = config['future_len']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout   ']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        self.use_map_img = config['use_map_image']
        self.use_map_lanes = config['use_map_lanes']
        self.past_T = config['past_len']
        self.max_points_per_lane = config['max_points_per_lane']
        self.max_num_roads = config['max_num_roads']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']

        self.road_pts_lin = nn.Sequential(
            init_(nn.Linear(self.map_attr, self.d_k)))
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(
            init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(self.num_queries_enc, self.d_k,
                                                  num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)

        output_query_provider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'],
            num_query_channels=self.d_k,
            init_scale=0.1,
        )

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        # self.map_positional_embedding = nn.parameter.Parameter(
        #     torch.zeros((1, self.max_points_per_lane * self.max_num_roads, self.d_k)), requires_grad=True
        # )

        self.perceiver_decoder = PerceiverDecoder(
            output_query_provider, self.d_k)

        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))

        self.output_model = nn.Sequential(
            init_(nn.Linear(self.d_k, 5 * self.T)))

        self.selu = nn.SELU(inplace=True)

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config
        pass

    def forward(self, out, gt, center_gt_final_valid_idx):
        return self.nll_loss_gmm_direct(out['predicted_probability'], out['predicted_trajectory'], gt,
                                        center_gt_final_valid_idx)

    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx,
                            pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0),
                            rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            timestamp_loss_weight (num_timestamps):
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
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(
            nearest_mode_idxs)  # (batch_size, 2)

        # (batch_size, num_timestamps, 5)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:,
                                                      :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(
                nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(
                nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(
                nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(
                nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * \
            torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
            (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp)
                    * gt_valid_mask).sum(dim=-1)

        loss_cls = (F.cross_entropy(input=pred_scores,
                    target=nearest_mode_idxs, reduction='none'))

        return (reg_loss + loss_cls).mean()
