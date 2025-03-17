import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from jarvis.transformers.wayformer.dataset import LSTMDataset
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
config_path = "config/trajectory_config.yaml"  # adjust if needed
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create datasets for training and validation using your LazyBaseDataset
train_dataset = LSTMDataset(config=config, is_validation=False)
print("Train dataset length:", len(train_dataset))
val_dataset = LSTMDataset(config=config, is_validation=True)

num_workers = 12
batch_size = 8

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=val_dataset.collate_fn
)

# Define a LightningModule for an LSTM-based trajectory predictor.
class LSTMTrajectoryPredictor(LightningModule):
    def __init__(self, config):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.config = config
        self.input_size = config['input_size']       # e.g., number of features, such as 7 ([x,y,z,heading,...])
        self.hidden_size = config['hidden_size']       # e.g., 128
        self.num_layers = config['num_layers']         # e.g., 2
        self.output_size = config['output_size']       # typically same as input_size
        self.past_len = config['past_len']             # length of past trajectory (input)
        self.future_len = config['future_len']         # length of future trajectory (target)

        # Encoder LSTM: processes past trajectory.
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # Decoder LSTM: predicts future trajectory.
        self.decoder = nn.LSTM(self.output_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.criterion = nn.MSELoss()

    def forward(self, past_traj, future_traj=None, teacher_forcing_ratio=0.5):
        """
        Args:
            past_traj (torch.Tensor): Tensor of shape (batch, past_len, input_size).
            future_traj (torch.Tensor, optional): Ground truth future trajectory 
                of shape (batch, future_len, output_size) for teacher forcing.
            teacher_forcing_ratio (float): Probability of using ground truth as next input.
        Returns:
            torch.Tensor: Predicted future trajectory of shape (batch, future_len, output_size).
        """
        batch_size = past_traj.size(0)
        # Encode the past trajectory.
        _, (h, c) = self.encoder(past_traj)  # h: (num_layers, batch, hidden_size)
        # Initialize decoder with the last observation from past_traj.
        decoder_input = past_traj[:, -1:, :]  # shape: (batch, 1, input_size)
        outputs = []
        for t in range(self.future_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)  # shape: (batch, 1, output_size)
            outputs.append(pred)
            # Teacher forcing: use ground truth with a certain probability.
            if future_traj is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = future_traj[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def training_step(self, batch, batch_idx):
        # Assume the collate_fn produces an 'input_dict' with keys 'obj_trajs' and 'center_gt_trajs'
        inputs = batch['input_dict']
        past_traj = inputs['obj_trajs'].float()    # shape: (batch, past_len, feature_dim)
        future_traj = inputs['center_gt_trajs'].float()  # shape: (batch, future_len, feature_dim)
        pred_future = self.forward(past_traj, future_traj, teacher_forcing_ratio=0.7)
        loss = self.criterion(pred_future, future_traj)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_dict']
        past_traj = inputs['obj_trajs'].float()
        future_traj = inputs['center_gt_trajs'].float()
        pred_future = self.forward(past_traj, future_traj, teacher_forcing_ratio=0.0)
        loss = self.criterion(pred_future, future_traj)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        return optimizer

# Create model instance and move to device.
model = LSTMTrajectoryPredictor(config)
model = model.to(device)

# Set up logging and checkpointing.
name = "lstm_trajectory"
logger = TensorBoardLogger("tb_logs", name=name)
checkpoint_dir = name + "_checkpoint/"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="lstm-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")

# Initialize the Trainer.
trainer = Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    max_epochs=config.get('max_epochs', 100),
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
    precision=16  # Enable mixed precision for faster training (optional)
)

# Train the model.
trainer.fit(model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        ckpt_path=latest_checkpoint)
