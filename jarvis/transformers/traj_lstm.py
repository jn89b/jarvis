import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from typing import Dict, Any, Tuple
from matplotlib import pyplot as plt

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
        self.fig_num = 0
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
        Computes the L2 loss over a trajectory.
        
        This loss function computes the L2 (Euclidean) distance between the predicted 
        trajectories and the ground truth for every mode, then averages these distances 
        over all timesteps, agents, modes, and the batch.
        
        Args:
            pred_params (torch.Tensor): shape (batch, num_agents, num_modes, future_len, output_dim)
            mode_probs (torch.Tensor): shape (batch, num_agents, num_modes) -- not used in this loss
            target (torch.Tensor): ground truth, shape (batch, num_agents, future_len, output_dim)
            
        Returns:
            torch.Tensor: scalar loss.
        """
        target_exp = target.unsqueeze(2)
        # Compute differences between predictions and target.
        # The resulting shape is (batch, num_agents, num_modes, future_len, output_dim)
        # Compute the squared differences
        sq_diff = (pred_params - target_exp) ** 2
        
        # Sum across ALL dimensions, giving a single scalar
        loss = sq_diff.sum()

        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        # Use only the first 3 dimensions (x, y, z) for the loss.
        future_traj = inputs['center_gt_trajs'].float()[:, :, :, 0:3]
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.5)
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


class SingleAgentLSTMTrajectoryPredictor(LightningModule):
    def __init__(self, config):
        """
        Args:
            config (dict): Contains model hyperparameters such as:
                - input_size: features in the past trajectory (e.g., 7)
                - hidden_size: hidden dimension for LSTM (e.g., 128)
                - num_layers: number of LSTM layers (e.g., 2)
                - output_dim: dimension of state to predict (e.g., 2 or 3)
                - num_modes: number of possible trajectories (e.g., 6)
                - past_len: length of past trajectory
                - future_len: prediction horizon (number of future timesteps)
                - learning_rate: optimizer learning rate
        """
        super(SingleAgentLSTMTrajectoryPredictor, self).__init__()
        self.config = config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.output_dim = config['output_dim']
        self.num_modes = config['num_modes']
        self.past_len = config['past_len']
        self.future_len = config['future_len']
        self.fig_num = 0

        # Encoder: processes past trajectory
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               dropout=0.1)
        # Decoder: generates predictions one timestep at a time.
        self.decoder = nn.LSTM(self.output_dim, self.hidden_size, self.num_layers, batch_first=True)
        # Fully-connected layer: for each timestep, outputs (num_modes * output_dim)
        self.fc = nn.Linear(self.hidden_size, self.num_modes * self.output_dim)
        # Mode probability predictor (using the last encoder hidden state)
        self.mode_prob_fc = nn.Linear(self.hidden_size, self.num_modes)

    def forward(self, batch: Dict[str, Any], teacher_forcing_ratio=0.5):
            """
            Args:
                batch (dict): Contains the key 'input_dict' with input tensors.
                    - 'obj_trajs': past trajectory tensor of shape (batch, past_len, input_size)
                    - 'center_gt_trajs': ground truth future trajectories of shape (batch, future_len, output_dim)
                teacher_forcing_ratio (float): probability of using ground truth as next input.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                - pred_params: predicted trajectories of shape (batch, num_modes, future_len, output_dim)
                - mode_probs: predicted mode probabilities of shape (batch, num_modes)
            """
            inputs = batch['input_dict']
            future_traj = inputs['center_gt_trajs'].float()  # (batch, future_len, output_dim)
            batch_size, num_agents, past_len, _ = inputs['obj_trajs'].size()

            past_traj = inputs['obj_trajs'].float().view(batch_size * num_agents, past_len, self.input_size)
            # Encode the past trajectory.
            _, (h, c) = self.encoder(past_traj)
            # Use the last timestep of past_traj as the initial input for the decoder.
            # Here, we take the first self.output_dim features.
            decoder_input = past_traj[:, -1:, :self.output_dim]  # shape: (batch, 1, output_dim)
            outputs = []

            for t in range(self.future_len):
                out, (h, c) = self.decoder(decoder_input, (h, c))
                # fc layer produces output of shape: (batch, 1, num_modes * output_dim)
                pred_t = self.fc(out)
                outputs.append(pred_t)
                decoder_input = pred_t[:, :, :3]  # shape: (batch, 1, 3)

            # Concatenate outputs along the time dimension.
            outputs = torch.cat(outputs, dim=1)  # shape: (batch, future_len, num_modes * output_dim)
            # Reshape into (batch, future_len, num_modes, output_dim)
            outputs = outputs.view(outputs.size(0), self.future_len, self.num_modes, self.output_dim)
            # Permute to (batch, num_modes, future_len, output_dim)
            pred_params = outputs.permute(0, 2, 1, 3)

            # Mode probabilities: use the last encoder hidden state.
            h_last = h[-1]  # shape: (batch, hidden_size)
            mode_logits = self.mode_prob_fc(h_last)  # shape: (batch, num_modes)
            mode_probs = torch.softmax(mode_logits, dim=-1)

            return pred_params, mode_probs

    def self_prediction_loss(self, pred_params, mode_probs, target):
        """
        Computes the L2 loss over a trajectory.
        
        Args:
            pred_params (torch.Tensor): shape (batch, num_modes, future_len, output_dim)
            mode_probs (torch.Tensor): shape (batch, num_modes) -- not used in this loss
            target (torch.Tensor): ground truth, shape (batch, future_len, output_dim)
            
        Returns:
            torch.Tensor: scalar loss.
        """
        #target_exp = target.unsqueeze(1)  # shape becomes (batch, 1, future_len, output_dim)
        # future_len = target.size(1)
        # sq_diff = (pred_params - target[:,:,:,0:3]) ** 2
        # loss = sq_diff.sum()
        mae_loss = nn.MSELoss()(pred_params, target[:,:,:,0:3])
        # # plot one trajectory
        fig, ax = plt.subplots()
        desired = target[0,0,:,0:3].detach().cpu().numpy()
        predicted = pred_params[0,0,:,0:3].detach().cpu().numpy()
        
        ax.plot(predicted[:, 0], predicted[:, 1], 'o--', label="Predicted",
                alpha=1.0)
        ax.plot(desired[:, 0], desired[:, 1], label="Ground Truth", linewidth=5)
        ax.legend()
        # save the figure
        plt.savefig(f"trajectory_{self.fig_num}.png")
        self.fig_num += 1
        
        return mae_loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        # Assuming ground truth uses the first output_dim (e.g., x, y, z) for the loss.
        future_traj = inputs['center_gt_trajs'].float() # shape: (batch, future_len, output_dim)
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.5)
        loss = self.self_prediction_loss(pred_params, mode_probs, future_traj)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        future_traj = inputs['center_gt_trajs'].float()  # shape: (batch, future_len, output_dim)
        pred_params, mode_probs = self.forward(batch, teacher_forcing_ratio=0.0)
        loss = self.self_prediction_loss(pred_params, mode_probs, future_traj)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        return optimizer


    
class LSTMEncoder(nn.Module):
    def __init__(self, 
                 input_size:int,
                 hidden_size:int,
                 num_layers:int,
                 bidirectional:bool=True,
                 dropout:float=0.1):
        """
        https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b
        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
        """
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=self.dropout)
        
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, input_size).
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - output: Tensor of shape (seq_len, batch, hidden_size).
                - (h_n, c_n): Tuple of hidden and cell states at the last time step.
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)
    
    def init_hidden(self, batch_size:int, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size (int): Size of the batch.
            device (torch.device): Device to store the tensors.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of hidden and cell states.
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c
    
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim,
                 output_size,
                 batch_size,
                 n_layers,
                 forecasting_horizon,
                 bidirectional=False,
                 dropout_p=0,):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.output_size = output_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.forecasting_horizon = forecasting_horizon

        self.lstm = nn.LSTM(hidden_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_p)

        self.out = nn.Linear(hidden_dim, output_size)

    def forward(self,
                decoder_input,
                encoder_hidden):

        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.forecasting_horizon):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_hidden[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, X, hidden):
        print(f'Decoder Before:, input: {X.shape}, h: {hidden[0].shape}, c: {hidden[1].shape}')
        output, hidden = self.lstm(X, hidden)
        print(f'Decoder After:, output: {output.shape}, h: {hidden[0].shape}, c: {hidden[1].shape}')
        output = self.out(output)
        print(f'Decoder Final:, output: {output.shape}, h: {hidden[0].shape}, c: {hidden[1].shape}')
        return output, hidden


class EncoderDecoderLSTM(LightningModule):
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
        super(EncoderDecoderLSTM, self).__init__()
        self.config = config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.output_dim = config['output_dim']
        self.num_modes = config['num_modes']
        self.past_len = config['past_len']
        self.future_len = config['future_len']
        
        self.encoder = LSTMEncoder(self.input_size, 
                                   self.hidden_size, 
                                   self.num_layers)
        self.decoder = DecoderLSTM(self.hidden_size, self.output_dim, 
                                   self.future_len, 
                                   self.num_layers, 
                                   self.future_len)
    

