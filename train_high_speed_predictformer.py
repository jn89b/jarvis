import numpy as np
import yaml
import torch
from jarvis.transformers.wayformer.dataset import BaseDataset, LazyBaseDataset
from jarvis.transformers.wayformer.predictformer import PredictFormer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
config_path = "config/high_speed_predictformer_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create separate datasets for training and validation
train_dataset = LazyBaseDataset(config=config, is_validation=False)
print("Train dataset length: ", len(train_dataset))
val_dataset = LazyBaseDataset(config=config, is_validation=True)

num_workers:int = 16
batch_size:int = 8
# Create DataLoaders for each dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,         # Shuffle training data
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,        # Usually don't shuffle validation data
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=val_dataset.collate_fn
)

# Load model configuration (if different from data config, adjust accordingly)
with open(config_path, 'r') as f:
    model_config = yaml.safe_load(f)

model = PredictFormer(model_config)
name = "high_speed_predictformer"
logger = TensorBoardLogger("tb_logs", name=name)

# Set up checkpointing
checkpoint_dir = name + "_checkpoint/"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="uavt-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f)
         for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")

# Initialize the Trainer (adjust parameters as needed)
trainer = Trainer(
    accelerator='gpu',  # Use 'gpu' if available
    devices=1,
    max_epochs=100,       # Specify a finite number of epochs
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
    precision=16          # Optionally, enable mixed precision for speed
)

# Train the model using both training and validation DataLoaders
trainer.fit(model, train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader, ckpt_path=latest_checkpoint)
