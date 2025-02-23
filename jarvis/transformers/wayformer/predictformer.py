import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import Dict, Any
# from unitraj.models.base_model.base_model import BaseModel
from jarvis.transformers.wayformer.base_modelV2 import BaseModelV2

from jarvis.transformers.wayformer.wayformer_utils import \
    (PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider)


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PredictFormer(BaseModelV2):
    def __init__(self, config):
        super(PredictFormer, self).__init__(config)
        self.config = config

        def init_(m): return init(m, nn.init.xavier_normal_,
                                  lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.fisher_information = None
        self.k_attr = config['num_agent_feature']
        self.d_k = config['hidden_size']
        self._M = config['max_num_agents']  # num agents without the ego-agent
        self.c = config['num_modes']
        self.T = config['future_len']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        self.past_T = config['past_len']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']

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

        self.perceiver_decoder = PerceiverDecoder(
            output_query_provider, self.d_k)

        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))

        # 7 is the number of
        # the GMM will output 7 parameters for each mode.
        # parameters are the following [x, y, z, log_std_x, log_std_y, log_std_z, rho_xy]
        self.num_parameters: int = 7
        self.output_model = nn.Sequential(
            init_(nn.Linear(self.d_k, 7 * self.T)))

        self.selu = nn.SELU(inplace=True)

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

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
        agents_tensor = agents_tensor.float()
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[
            :, :, :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)

        mixed_input_features = torch.concat(
            [agents_emb], dim=1)
        mixed_input_masks = torch.concat(
            [opps_masks_agents.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(
            mixed_input_features, mixed_input_masks)

        # Wazformer-Ego Decoding

        out_seq = self.perceiver_decoder(context)
        # TODO: Access the standard deviations and correlation coefficients
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

    def forward(self, batch: Dict[str, Any]):
        """
        Args:
            batch: A dictionary containing the following
                input_dict: A dictionary containing the following
                    obj_trajs: A tensor of shape [B, T, M, k_attr] where M is the number of agents in the scene.
                    obj_trajs_mask: A tensor of shape [B, T, M] where M is the number of agents in the scene.
        T

        """
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask = inputs['obj_trajs'], inputs['obj_trajs_mask']
        agents_in = agents_in.squeeze()
        agents_mask = agents_mask.squeeze()
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                          -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
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
        ground_truth = ground_truth.squeeze()
        loss = self.criterion(output, ground_truth,
                              inputs['center_gt_final_valid_idx'])
        # output['dataset_name'] = inputs['dataset_name']
        output['predicted_probability'] = F.softmax(
            output['predicted_probability'], dim=-1)
        return output, loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.config['learning_rate'], eps=0.0001)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0002, steps_per_epoch=1, epochs=500,
                                                        pct_start=0.02, div_factor=100.0, final_div_factor=10)

        return [optimizer], [scheduler]


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



        Extended GMM Loss for Motion Transformer (MTR) with 3D trajectories (x, y, z).

        Args:
            pred_scores (Tensor): Shape (batch_size, num_modes).
            pred_trajs (Tensor): Shape (batch_size, num_modes, num_timestamps, features).
                For use_square_gmm: features = 4  (x, y, z, log_std)
                For full GMM: features = 7  (x, y, z, log_std_x, log_std_y, log_std_z, rho_xy)
            gt_trajs (Tensor): Shape (batch_size, num_timestamps, 4).
                The first three channels are x, y, z positions and the last channel is the validity mask.
            center_gt_final_valid_idx: (Not used in this simplified example)
            pre_nearest_mode_idxs (Tensor, optional): Pre-computed nearest mode indices.
            timestamp_loss_weight (Tensor, optional): Weight for each timestamp.
            use_square_gmm (bool, optional): Whether to use the square GMM formulation.
            log_std_range (tuple, optional): Clipping range for log standard deviations.
            rho_limit (float, optional): Limit for correlation coefficient.

        """
        # if use_square_gmm:
        #     assert pred_trajs.shape[-1] == 3
        # else:
        #     assert pred_trajs.shape[-1] == 5

        # if use_square_gmm:
        #     assert pred_trajs.shape[-1] == 4  # now x, y, z, log_std_z
        # else:
        #     # e.g., x, y, z, log_std₁, log_std₂, log_std₃, rho (or more if using full correlations)
        #     assert pred_trajs.shape[-1] == 7

        # batch_size = pred_trajs.shape[0]

        # gt_valid_mask = gt_trajs[..., -1]

        # if pre_nearest_mode_idxs is not None:
        #     nearest_mode_idxs = pre_nearest_mode_idxs
        # else:
        #     # distance = (pred_trajs[:, :, :, 0:2] -
        #     #             gt_trajs[:, None, :, :2]).norm(dim=-1)
        #     # distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        #     distance = (pred_trajs[:, :, :, 0:3] -
        #                 gt_trajs[:, None, :, 0:3]).norm(dim=-1)
        #     distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
        #     nearest_mode_idxs = distance.argmin(dim=-1)

        # nearest_mode_bs_idxs = torch.arange(batch_size).type_as(
        #     nearest_mode_idxs)  # (batch_size, 2)

        # # (batch_size, num_timestamps, 5)
        # # (batch_size, num_timestamps, features)
        # nearest_trajs = pred_trajs[torch.arange(batch_size), nearest_mode_idxs]
        # # now includes x, y, and z differences
        # res_trajs = gt_trajs[..., :3] - nearest_trajs[..., :3]
        # dx = res_trajs[:, :, 0]
        # dy = res_trajs[:, :, 1]
        # dz = res_trajs[:, :, 2]

        # if use_square_gmm:
        #     log_std1 = log_std2 = torch.clip(
        #         nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        #     std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
        #     rho = torch.zeros_like(log_std1)
        # else:
        #     log_std1 = torch.clip(
        #         nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        #     log_std2 = torch.clip(
        #         nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        #     std1 = torch.exp(log_std1)  # (0.2m to 150m)
        #     std2 = torch.exp(log_std2)  # (0.2m to 150m)
        #     rho = torch.clip(
        #         nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        # gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        # if timestamp_loss_weight is not None:
        #     gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # # -log(a^-1 * e^b) = log(a) - b
        # reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * \
        #     torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        # reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
        #     (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
        #         std1 * std2))  # (batch_size, num_timestamps)

        # reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp)
        #             * gt_valid_mask).sum(dim=-1)

        # loss_cls = (F.cross_entropy(input=pred_scores,
        #                             target=nearest_mode_idxs, reduction='none'))

        # return (reg_loss + loss_cls).mean()

        if use_square_gmm:
            # Expect 4 features: [x, y, z, log_std]
            assert pred_trajs.shape[-1] == 4, f"Expected 4 features for square GMM, got {pred_trajs.shape[-1]}"
        else:
            # Expect 7 features: [x, y, z, log_std_x, log_std_y, log_std_z, rho_xy]
            assert pred_trajs.shape[-1] == 7, f"Expected 7 features for full GMM, got {pred_trajs.shape[-1]}"

        batch_size = pred_trajs.shape[0]
        # Assume the last channel in gt_trajs is a validity mask for each timestamp.
        # shape: (batch_size, num_timestamps)
        gt_valid_mask = gt_trajs[..., -1]

        # --- Mode Selection ---
        # Compute Euclidean distance in 3D (x, y, z) between predicted trajectories and ground truth.
        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            # (B, num_modes, T)
            distance = (pred_trajs[:, :, :, 0:3] -
                        gt_trajs[:, None, :, 0:3]).norm(dim=-1)
            # Weight distance by the validity mask and sum over timesteps.
            # (B, num_modes)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
            nearest_mode_idxs = distance.argmin(dim=-1)  # (B,)

        nearest_mode_bs_idxs = torch.arange(
            batch_size).to(nearest_mode_idxs.device)
        # Select the nearest trajectory for each sample: shape (B, T, features)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]

        # --- Residual Computation ---
        # Compute the difference between ground truth and prediction for x, y, and z.
        res_trajs = gt_trajs[..., :3] - nearest_trajs[..., :3]  # (B, T, 3)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]
        dz = res_trajs[:, :, 2]

        # --- Negative Log-Likelihood (NLL) Loss Computation ---
        if use_square_gmm:
            # For square GMM, we assume one shared uncertainty for all three dimensions.
            log_std = torch.clip(
                nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std = torch.exp(log_std)
            # For x,y, assume independence (rho = 0).
            rho = torch.zeros_like(log_std)
            # 2D component for x and y.
            reg_gmm_log_coefficient_xy = log_std + \
                log_std + 0.5 * torch.log(1 - rho ** 2)
            reg_gmm_exp_xy = (0.5 / (1 - rho ** 2)) * \
                ((dx ** 2 + dy ** 2) / (std ** 2))
            # 1D component for z.
            reg_gmm_log_coefficient_z = log_std
            reg_gmm_exp_z = 0.5 * (dz / std) ** 2

            reg_loss = ((reg_gmm_log_coefficient_xy + reg_gmm_exp_xy +
                         reg_gmm_log_coefficient_z + reg_gmm_exp_z) * gt_valid_mask).sum(dim=-1)
        else:
            # For full GMM, we model x and y jointly with a correlation, and treat z independently.
            log_std_x = torch.clip(
                nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            log_std_y = torch.clip(
                nearest_trajs[:, :, 4], min=log_std_range[0], max=log_std_range[1])
            std_x = torch.exp(log_std_x)
            std_y = torch.exp(log_std_y)
            rho = torch.clip(
                nearest_trajs[:, :, 6], min=-rho_limit, max=rho_limit)
            reg_gmm_log_coefficient_xy = log_std_x + \
                log_std_y + 0.5 * torch.log(1 - rho ** 2)
            reg_gmm_exp_xy = (0.5 / (1 - rho ** 2)) * (
                (dx ** 2) / (std_x ** 2) + (dy ** 2) /
                (std_y ** 2) - 2 * rho * dx * dy / (std_x * std_y)
            )
            # For the z component, use an independent uncertainty (log_std_z is at index 5).
            log_std_z = torch.clip(
                nearest_trajs[:, :, 5], min=log_std_range[0], max=log_std_range[1])
            std_z = torch.exp(log_std_z)
            reg_gmm_log_coefficient_z = log_std_z
            reg_gmm_exp_z = 0.5 * (dz / std_z) ** 2

            reg_loss = ((reg_gmm_log_coefficient_xy + reg_gmm_exp_xy +
                         reg_gmm_log_coefficient_z + reg_gmm_exp_z) * gt_valid_mask).sum(dim=-1)

        # Apply optional timestamp loss weighting if provided.
        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # --- Classification Loss ---
        # Cross-entropy loss on the predicted mode scores encourages the model to choose the mode with minimal NLL.
        loss_cls = F.cross_entropy(
            input=pred_scores, target=nearest_mode_idxs, reduction='none')

        # Combine regression (NLL) loss and classification loss, then average over the batch.
        return (reg_loss + loss_cls).mean()
