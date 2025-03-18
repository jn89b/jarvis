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
# Create Dummy DataLoaders
# -------------------------
#dummy_dataset = DummyTrajectoryDataset(num_samples, config['num_agents'], config['past_len'], config['future_len'], config['input_size'], config['output_dim'])
# Load configuration
config_path = "config/lstm_config.yaml"  # adjust if needed
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

batch_size = 6
num_workers = 6
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
