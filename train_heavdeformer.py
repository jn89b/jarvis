import yaml
import os
import matplotlib.pyplot as plt


from typing import Dict, Any, List
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from jarvis.transformers.evaderformer2 import HEvadrFormer
# from jarvis.datasets.base_dataset import PlanTDataset
from jarvis.datasets.base_dataset import HATDataset

# Load the dataset configuration
data_config = "config/data_config.yaml"
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)
dataset = HATDataset(config=data_config, is_validation=False)
dataloader = DataLoader(dataset, batch_size=5,
                        shuffle=True, collate_fn=dataset.collate_fn)
valdataset = HATDataset(config=data_config, is_validation=True)
valdataloader = DataLoader(valdataset, batch_size=5,
                           shuffle=True, collate_fn=dataset.collate_fn)

model_config: Dict[str, Any] = {}
model = HEvadrFormer(model_config)
name = "havt"
logger = TensorBoardLogger("tb_logs", name=name)

# Checkpoint callback
checkpoint_dir = name+"_checkpoint/"
# checkpoint_dir = "evader_former_checkpoint/"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="havt-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min"
)

# test a batch
for batch in dataloader:
    print(batch)
    model(batch)
    break

# Check if there's an existing checkpoint to resume from

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
    max_epochs=2000,
    logger=logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Train the model, resuming from the latest checkpoint if it exists
trainer.fit(model, train_dataloaders=dataloader,
            val_dataloaders=dataloader, ckpt_path=latest_checkpoint)
