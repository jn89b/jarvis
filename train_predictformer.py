import numpy as np
import unittest
import yaml
import torch

from jarvis.transformers.wayformer.dataset import BaseDataset
from jarvis.transformers.wayformer.predictformer import PredictFormer
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_config = "config/predictformer_config.yaml"
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)


dataset = BaseDataset(
    config=data_config,
    num_samples=50)

dataloader: DataLoader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

model_config: str = "config/predictformer_config.yaml"
with open(model_config, 'r') as f:
    model_config = yaml.safe_load(f)

model: PredictFormer = PredictFormer(model_config)
name = "predictformer"
logger = TensorBoardLogger("tb_logs", name=name)

# Check if there's an existing checkpoint to resume from
checkpoint_dir = name+"_checkpoint/"
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
        print(
            f"Resuming training from checkpoint: {latest_checkpoint}")

# Initialize the Trainer
trainer = Trainer(
    devices=1,
    max_epochs=2000,
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Train the model, resuming from the latest checkpoint if it exists
trainer.fit(model, train_dataloaders=dataloader,
            val_dataloaders=dataloader, ckpt_path=latest_checkpoint)
