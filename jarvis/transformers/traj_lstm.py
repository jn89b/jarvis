import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from typing import Dict, Any

class MultiAgentLSTMTrajectoryPredictor(LightningModule):
    def __init__(self, config):
        """
        Args:
            config (dict): Contains model hyperparameters such as:
                - input_size: features in the past trajectory (e.g., 7)
                - hidden_size: hidden dimension for LSTM (e.g., 128)
                - num_layers: number of LSTM layers (e.g., 2)
                - output_dim: dimension of state to predict (e.g., 2 or 3)
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
        # Fully-connected layer: for each timestep, outputs (num_modes * output_dim)
        self.fc = nn.Linear(self.hidden_size, self.num_modes * self.output_dim)
        # Mode probability predictor (using the last encoder hidden state)
        self.mode_prob_fc = nn.Linear(self.hidden_size, self.num_modes)

    def forward(self, batch: Dict[str,Any], teacher_forcing_ratio=0.5):
        """
        Args:
            batch (dict): Contains the key 'input_dict' with input tensors.
            teacher_forcing_ratio (float): probability of using ground truth as next input.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - pred_params: predicted trajectories of shape (batch, num_agents, num_modes, future_len, output_dim)
              - mode_probs: predicted mode probabilities of shape (batch, num_agents, num_modes)
        """
        inputs = batch['input_dict']
        future_traj = inputs['center_gt_trajs'].float()
        batch_size, num_agents, past_len, _ = inputs['obj_trajs'].size()
        past_traj = inputs['obj_trajs'].float().view(batch_size * num_agents, past_len, self.input_size)
        
        # Encode the past trajectory.
        _, (h, c) = self.encoder(past_traj)
        # Use last timestep of past_traj as initial input for the decoder.
        decoder_input = past_traj[:, -1:, :self.output_dim]
        outputs = []
        for t in range(self.future_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            # Predict all modes: (batch*num_agents, 1, num_modes * output_dim)
            pred_t = self.fc(out)
            outputs.append(pred_t)
            # Teacher forcing: use ground truth with some probability.
            if future_traj is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Extract ground truth for the current timestep.
                teacher_input = future_traj[:, :, t:t+1, 0:3]  # adjust if output_dim differs
                teacher_input = teacher_input.view(batch_size * num_agents, 1, self.output_dim)
                decoder_input = teacher_input
            else:
                # Use the predicted mean of the first mode.
                pred_t_reshaped = pred_t.view(batch_size * num_agents, self.num_modes, self.output_dim)
                decoder_input = pred_t_reshaped[:, 0, :].unsqueeze(1)
        # Concatenate outputs along the time dimension.
        outputs = torch.cat(outputs, dim=1)  # shape: (batch*num_agents, future_len, num_modes * output_dim)
        outputs = outputs.view(batch_size * num_agents, self.future_len, self.num_modes, self.output_dim)
        # Permute to shape: (batch*num_agents, num_modes, future_len, output_dim)
        pred_params = outputs.permute(0, 2, 1, 3)
        # Reshape back to (batch, num_agents, num_modes, future_len, output_dim)
        pred_params = pred_params.view(batch_size, num_agents, self.num_modes, self.future_len, self.output_dim)
        
        # Mode probabilities: use the last encoder hidden state for each agent.
        h_last = h[-1]  # shape: (batch*num_agents, hidden_size)
        mode_logits = self.mode_prob_fc(h_last)  # shape: (batch*num_agents, num_modes)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        mode_probs = mode_probs.view(batch_size, num_agents, self.num_modes)
        
        return pred_params, mode_probs

    def self_prediction_loss(self, pred_params, mode_probs, target):
        """
        Computes a best-of-many self-prediction loss.
        For each agent, we select the mode with the minimum L2 error compared to the ground truth trajectory,
        then average these errors over all agents and the batch.
        
        Args:
            pred_params (torch.Tensor): shape (batch, num_agents, num_modes, future_len, output_dim)
            mode_probs (torch.Tensor): shape (batch, num_agents, num_modes) -- not used in this loss
            target (torch.Tensor): ground truth, shape (batch, num_agents, future_len, output_dim)
        Returns:
            torch.Tensor: scalar loss.
        """
        # Expand target to add a mode dimension: (batch, num_agents, 1, future_len, output_dim)
        target_exp = target.unsqueeze(2)
        # Compute squared errors for each mode over timesteps and output dimensions.
        errors = (pred_params - target_exp) ** 2  # shape: (batch, num_agents, num_modes, future_len, output_dim)
        # Sum over output_dim and average over future timesteps.
        errors = errors.sum(dim=-1).mean(dim=-1)   # shape: (batch, num_agents, num_modes)
        # Select the best (i.e., minimum) error over the modes for each agent.
        best_error, _ = torch.min(errors, dim=2)  # shape: (batch, num_agents)
        # Average the best errors over agents and batch.
        loss = best_error.mean()
        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        # Use only the first 3 dimensions (x, y, z) for the loss.
        future_traj = inputs['center_gt_trajs'].float()[:, :, :, 0:3]
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.1)
        loss = self.self_prediction_loss(pred_params, mode_probs, future_traj)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        future_traj = inputs['center_gt_trajs'].float()[:, :, :, 0:3]
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.0)
        loss = self.self_prediction_loss(pred_params, mode_probs, future_traj)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        return optimizer
