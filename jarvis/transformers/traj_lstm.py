import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from typing import Dict, Any
class MultiAgentLSTMTrajectoryPredictor(LightningModule):
    def __init__(self, config):
        """
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489734
        https://stackoverflow.com/questions/65269119/encoder-decoder-for-trajectory-prediction
        Args:
            config (dict): Should contain:
                - input_size: features in the past trajectory (e.g., 7)
                - hidden_size: hidden dimension for LSTM (e.g., 128)
                - num_layers: number of LSTM layers (e.g., 2)
                - output_dim: dimension of state to predict (e.g., 2 for [x, y])
                - num_modes: number of possible trajectories per agent (e.g., 6)
                - past_len: length of past trajectory
                - future_len: prediction horizon (number of future timesteps)
                - learning_rate: optimizer learning rate
        """
        super(MultiAgentLSTMTrajectoryPredictor, self).__init__()
        self.config = config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.output_dim = config['output_dim']
        self.num_modes = config['num_modes']
        self.past_len = config['past_len']
        self.future_len = config['future_len']

        # Encoder: processes past trajectory (for each agent)
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # Decoder: generates predictions one timestep at a time.
        self.decoder = nn.LSTM(self.output_dim, self.hidden_size, self.num_layers, batch_first=True)
        # FC layer: for each timestep, outputs (num_modes * output_dim)
        self.fc = nn.Linear(self.hidden_size, self.num_modes * self.output_dim)
        # Mode probability predictor (using the last encoder hidden state)
        self.mode_prob_fc = nn.Linear(self.hidden_size, self.num_modes)

    def forward(self, 
                batch:Dict[str,Any], teacher_forcing_ratio=0.5):
        """
        Args:
            past_traj (torch.Tensor): shape (batch, num_agents, past_len, input_size)
            future_traj (torch.Tensor, optional): shape (batch, num_agents, future_len, output_dim)
            teacher_forcing_ratio (float): probability of using ground truth as next input.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - Predicted parameters with shape (batch, num_agents, num_modes, future_len, output_dim)
              - Mode probabilities with shape (batch, num_agents, num_modes)
        """
        inputs = batch['input_dict']
        future_traj = inputs['center_gt_trajs'].float()
        batch_size, num_agents, past_len, _ = inputs['obj_trajs'].size()
        past_traj = inputs['obj_trajs'].float()
        # Merge batch and agent dimensions: (batch*num_agents, past_len, input_size)
        past_traj = past_traj.view(batch_size * num_agents, past_len, self.input_size)
        # Encode the past trajectory.
        _, (h, c) = self.encoder(past_traj)  # h: (num_layers, batch*num_agents, hidden_size)
        # Use last timestep of past_traj as initial input for the decoder.
        decoder_input = past_traj[:, -1:, :self.output_dim]  # (batch*num_agents, 1, output_dim)
        outputs = []
        for t in range(self.future_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            # Output for all modes: (batch*num_agents, 1, num_modes * output_dim)
            pred_t = self.fc(out)
            outputs.append(pred_t)
            # Teacher forcing: if future_traj provided, sometimes use ground truth.
            if future_traj is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Extract ground truth for current timestep:
                teacher_input = future_traj[:, :, t:t+1, 0:3]  # (batch, num_agents, 1, output_dim)
                teacher_input = teacher_input.view(batch_size * num_agents, 1, self.output_dim)
                decoder_input = teacher_input
            else:
                # Use the predicted mean of the first mode.
                pred_t_reshaped = pred_t.view(batch_size * num_agents, self.num_modes, self.output_dim)
                decoder_input = pred_t_reshaped[:, 0, :].unsqueeze(1)
        # Concatenate outputs along the time dimension:
        outputs = torch.cat(outputs, dim=1)  # (batch*num_agents, future_len, num_modes * output_dim)
        outputs = outputs.view(batch_size * num_agents, self.future_len, self.num_modes, self.output_dim)
        # Permute to shape (batch*num_agents, num_modes, future_len, output_dim)
        pred_params = outputs.permute(0, 2, 1, 3)
        # Reshape back to (batch, num_agents, num_modes, future_len, output_dim)
        pred_params = pred_params.view(batch_size, num_agents, self.num_modes, self.future_len, self.output_dim)
        
        # Mode probabilities: use the last encoder hidden state for each agent.
        h_last = h[-1]  # (batch*num_agents, hidden_size)
        mode_logits = self.mode_prob_fc(h_last)  # (batch*num_agents, num_modes)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        mode_probs = mode_probs.view(batch_size, num_agents, self.num_modes)
        
        return pred_params, mode_probs

    def min_of_n_loss(self, pred_params, target):
        """
        Computes a "min-of-N" L2 loss.
        Args:
            pred_params (torch.Tensor): shape (batch, num_agents, num_modes, future_len, output_dim)
            target (torch.Tensor): shape (batch, num_agents, future_len, output_dim)
        Returns:
            torch.Tensor: scalar loss.
        """
        # Expand target: (batch, num_agents, 1, future_len, output_dim)
        target_exp = target.unsqueeze(2)
        # Compute L2 errors per mode, then average over timesteps and dimensions.
        errors = (pred_params - target_exp) ** 2
        errors = errors.sum(dim=-1)   # (batch, num_agents, num_modes, future_len)
        errors = errors.mean(dim=-1)  # (batch, num_agents, num_modes)
        # For each agent in each sample, choose the mode with the smallest error.
        min_errors, _ = errors.min(dim=-1)  # (batch, num_agents)
        loss = min_errors.mean()
        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        # past_traj = inputs['obj_trajs'].float()        # expected shape: (batch, past_len, num_agents, input_size) or (batch, num_agents, past_len, input_size)
        future_traj = inputs['center_gt_trajs'].float()  # expected shape: (batch, future_len, num_agents, output_dim) or (batch, num_agents, future_len, output_dim)
        future_traj = future_traj[:, :, :, 0:3]  # Keep only x, y, z coordinates
        # Ensure that past_traj is of shape (batch, num_agents, past_len, input_size)
        # and future_traj is (batch, num_agents, future_len, output_dim).
        # (Transpose dimensions if necessary.)
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.7)
        loss = self.min_of_n_loss(pred_params, future_traj)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        # past_traj = inputs['obj_trajs'].float()
        future_traj = inputs['center_gt_trajs'].float()
        future_traj = future_traj[:, :, :, 0:3]  # Keep only x, y, z coordinates
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.0)
        loss = self.min_of_n_loss(pred_params, future_traj)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        return optimizer
