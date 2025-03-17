import os
import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from jarvis.transformers.wayformer.dataset import LSTMDataset
from jarvis.transformers.traj_lstm import MultiAgentLSTMTrajectoryPredictor

# -------------------------
# Dummy Dataset Definition
# # -------------------------
# class DummyTrajectoryDataset(Dataset):
#     def __init__(self, num_samples, num_agents, past_len, future_len, input_size, output_dim):
#         """
#         Args:
#             num_samples (int): Number of samples in the dataset.
#             num_agents (int): Number of agents per sample.
#             past_len (int): Number of timesteps in the past trajectory.
#             future_len (int): Number of timesteps in the future trajectory.
#             input_size (int): Dimensionality of each past timestep (e.g., 7 features).
#             output_dim (int): Dimensionality of each future timestep (e.g., 2 for [x,y]).
#         """
#         self.num_samples = num_samples
#         self.num_agents = num_agents
#         self.past_len = past_len
#         self.future_len = future_len
#         self.input_size = input_size
#         self.output_dim = output_dim

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         # Generate random past trajectories and future trajectories for each agent.
#         # Past: shape (num_agents, past_len, input_size)
#         past_traj = torch.randn(self.num_agents, self.past_len, self.input_size)
#         # Future: shape (num_agents, future_len, output_dim)
#         future_traj = torch.randn(self.num_agents, self.future_len, self.output_dim)
#         return {
#             'input_dict': {
#                 'obj_trajs': past_traj,
#                 'center_gt_trajs': future_traj
#             }
#         }

# # -------------------------
# # Configuration Dictionary
# # -------------------------
# config = {
#     'input_size': 7,      # e.g., features: [x, y, z, heading, ...]
#     'hidden_size': 32,    # hidden dimension for LSTM
#     'num_layers': 1,      # number of LSTM layers
#     'output_dim': 7,      # predicting [x, y] positions
#     'num_modes': 6,       # number of predicted trajectories (modes) per agent
#     'past_len': 21,       # length of past trajectory (timesteps)
#     'future_len': 60,      # prediction horizon (timesteps)
#     'learning_rate': 0.001,
#     'max_epochs': 150,
#     'num_agents': 3       # number of agents per sample
# }

# # -------------------------
# # Lightning Module Definition
# # -------------------------
# class MultiAgentLSTMTrajectoryPredictor(LightningModule):
#     def __init__(self, config):
#         super(MultiAgentLSTMTrajectoryPredictor, self).__init__()
#         self.config = config
#         self.input_size = config['input_size']
#         self.hidden_size = config['hidden_size']
#         self.num_layers = config['num_layers']
#         self.output_dim = config['output_dim']
#         self.num_modes = config['num_modes']
#         self.past_len = config['past_len']
#         self.future_len = config['future_len']
        
#         # Encoder: processes past trajectory.
#         self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
#         # Decoder: generates predictions for each future timestep.
#         self.decoder = nn.LSTM(self.output_dim, self.hidden_size, self.num_layers, batch_first=True)
#         # Fully connected layer outputs predictions for all modes at once.
#         # For each timestep, output: num_modes * output_dim
#         self.fc = nn.Linear(self.hidden_size, self.num_modes * self.output_dim)
#         # Mode probability predictor.
#         self.mode_prob_fc = nn.Linear(self.hidden_size, self.num_modes)
    
#     def forward(self, past_traj, future_traj=None, teacher_forcing_ratio=0.5):
#         """
#         Args:
#             past_traj: (batch, num_agents, past_len, input_size)
#             future_traj: (batch, num_agents, future_len, output_dim) or None
#             teacher_forcing_ratio: probability to use ground truth as next input.
#         Returns:
#             Tuple:
#               - pred_params: (batch, num_agents, num_modes, future_len, output_dim)
#               - mode_probs: (batch, num_agents, num_modes)
#         """
#         batch_size, num_agents, _, _ = past_traj.size()
#         # Merge batch and agent dimensions: shape (batch*num_agents, past_len, input_size)
#         past_traj = past_traj.view(batch_size * num_agents, self.past_len, self.input_size)
#         _, (h, c) = self.encoder(past_traj)
#         # Initial decoder input: last timestep from past, using only first output_dim features.
#         decoder_input = past_traj[:, -1:, :self.output_dim]  # (batch*num_agents, 1, output_dim)
#         outputs = []
#         for t in range(self.future_len):
#             out, (h, c) = self.decoder(decoder_input, (h, c))
#             # Output: (batch*num_agents, 1, num_modes * output_dim)
#             pred_t = self.fc(out)
#             outputs.append(pred_t)
#             if future_traj is not None and torch.rand(1).item() < teacher_forcing_ratio:
#                 teacher_input = future_traj.view(batch_size * num_agents, self.future_len, self.output_dim)[:, t:t+1, :]
#                 decoder_input = teacher_input
#             else:
#                 pred_t_reshaped = pred_t.view(batch_size * num_agents, self.num_modes, self.output_dim)
#                 decoder_input = pred_t_reshaped[:, 0, :].unsqueeze(1)
#         outputs = torch.cat(outputs, dim=1)  # (batch*num_agents, future_len, num_modes * output_dim)
#         outputs = outputs.view(batch_size * num_agents, self.future_len, self.num_modes, self.output_dim)
#         outputs = outputs.permute(0, 2, 1, 3)  # (batch*num_agents, num_modes, future_len, output_dim)
#         pred_params = outputs.view(batch_size, num_agents, self.num_modes, self.future_len, self.output_dim)
        
#         h_last = h[-1]  # (batch*num_agents, hidden_size)
#         mode_logits = self.mode_prob_fc(h_last)  # (batch*num_agents, num_modes)
#         mode_probs = torch.softmax(mode_logits, dim=-1)
#         mode_probs = mode_probs.view(batch_size, num_agents, self.num_modes)
        
#         return pred_params, mode_probs

#     def min_of_n_loss(self, pred_params, target):
#         """
#         Computes "min-of-N" L2 loss.
#         Args:
#             pred_params: (batch, num_agents, num_modes, future_len, output_dim)
#             target: (batch, num_agents, future_len, output_dim)
#         Returns:
#             Scalar loss.
#         """
#         target_exp = target.unsqueeze(2)  # (batch, num_agents, 1, future_len, output_dim)
#         errors = (pred_params - target_exp) ** 2
#         errors = errors.sum(dim=-1)  # (batch, num_agents, num_modes, future_len)
#         errors = errors.mean(dim=-1)  # (batch, num_agents, num_modes)
#         min_errors, _ = errors.min(dim=-1)  # (batch, num_agents)
#         loss = min_errors.mean()
#         return loss

#     def training_step(self, batch, batch_idx):
#         inputs = batch['input_dict']
#         past_traj = inputs['obj_trajs'].float()         # (batch, num_agents, past_len, input_size)
#         future_traj = inputs['center_gt_trajs'].float()   # (batch, num_agents, future_len, output_dim)
#         pred_params, mode_probs = self.forward(past_traj, future_traj, teacher_forcing_ratio=0.7)
#         loss = self.min_of_n_loss(pred_params, future_traj)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs = batch['input_dict']
#         past_traj = inputs['obj_trajs'].float()
#         future_traj = inputs['center_gt_trajs'].float()
#         pred_params, mode_probs = self.forward(past_traj, future_traj, teacher_forcing_ratio=0.0)
#         loss = self.min_of_n_loss(pred_params, future_traj)
#         self.log('val_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
#         return optimizer

# -------------------------
# Create Dummy DataLoaders
# -------------------------
#dummy_dataset = DummyTrajectoryDataset(num_samples, config['num_agents'], config['past_len'], config['future_len'], config['input_size'], config['output_dim'])
# Load configuration
config_path = "config/lstm_config.yaml"  # adjust if needed
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

batch_size = 6
num_workers = 8
# Create datasets for training and validation using your LazyBaseDataset
train_dataset = LSTMDataset(config=config, is_validation=False)
val_dataset = LSTMDataset(config=config, is_validation=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, 
                              collate_fn=train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, 
                            collate_fn=val_dataset.collate_fn)

# -------------------------
# Set up Logging & Checkpointing
# -------------------------
name = "lstm_multi_trajectory"
logger = TensorBoardLogger("tb_logs", name=name)
checkpoint_dir = name + "_checkpoint/"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="lstm_multi_trajectory-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    ckpt_files = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
                        key=os.path.getmtime)
    if ckpt_files:
        latest_checkpoint = ckpt_files[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")

# -------------------------
# Initialize Trainer and Train Model
# -------------------------
trainer = Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    max_epochs=config.get('max_epochs', 150),
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
    precision=16
)

model = MultiAgentLSTMTrajectoryPredictor(config)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=latest_checkpoint)
