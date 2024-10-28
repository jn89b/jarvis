from pytorch_lightning import Trainer
import yaml
import torch
from jarvis.datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from jarvis.transformers.evadeformer import EvadeFormer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Load the dataset configuration
data_config = "config/data_config.yaml"
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset and dataloader
batch_size = 5
dataset = BaseDataset(config=data_config, is_validation=False)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=dataset.collate_fn)

# Load model configuration
model_config = 'config/data_config.yaml'
with open(model_config, 'r') as f:
    model_config = yaml.safe_load(f)

# Initialize your model
model = EvadeFormer(model_config)

# TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="evadeformer")

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="evadeformer-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)

# Check if there's an existing checkpoint to resume from
checkpoint_dir = "checkpoints/"
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

# Initialize the Trainer
trainer = Trainer(
    devices=1,
    max_epochs=800,
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Train the model, resuming from the latest checkpoint if it exists
trainer.fit(model, train_dataloaders=dataloader,
            val_dataloaders=dataloader, ckpt_path=latest_checkpoint)
