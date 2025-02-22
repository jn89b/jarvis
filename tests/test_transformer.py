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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTransformer(unittest.TestCase):

    def setUp(self):

        # Load the dataset configuration
        data_config = "config/predictformer_config.yaml"

        with open(data_config, 'r') as f:
            data_config = yaml.safe_load(f)

        self.dataset = BaseDataset(
            config=data_config,
            num_samples=1
        )

        self.dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.dataset.collate_fn
        )
        # make sure data

    def test_dataset(self):
        """
        Idiot checks to make sure the dataset is not None
        and that are we are able to access the data_path and open the correct
        json file
        """
        self.assertTrue(self.dataset is not None)
        # check the data_path attribute
        data_path: str = self.dataset.data_path
        self.assertTrue(data_path is not None)
        # self.dataset.segment_data()

    def test_loading(self):
        """
        Idiot checks to make sure we are loading the data correctly
        to feed into the model
        """

        self.assertTrue(self.dataloader is not None)
        # check the batch size

        # for i, batch in enumerate(self.dataloader):
        #     print(batch)
        #     # make sure the input_dictionary values are torch tensors
        #     input_dict = batch['input_dict']
        #     for k, v in input_dict.items():
        #         print(k, v)
        # self.assertTrue(isinstance(batch['input_dict'], torch.Tensor))

    def test_training(self):
        """
        Idiot checks to make sure we are able to train the model
        """
        model_config = "config/predictformer_config.yaml"
        with open(model_config, 'r') as f:
            model_config = yaml.safe_load(f)

        self.model: PredictFormer = PredictFormer(model_config)
        name = "predictformer"
        logger = TensorBoardLogger("tb_logs", name=name)

        # Checkpoint callback
        checkpoint_dir = name+"_checkpoint/"
        # checkpoint_dir = "evader_former_checkpoint/"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename="uavt-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,
            mode="min"
        )

        # test a batch
        for batch in self.dataloader:
            print(batch)
            output, loss = self.model(batch)

        # Check if there's an existing checkpoint to resume from

        # latest_checkpoint = None
        # if os.path.exists(checkpoint_dir):
        #     checkpoint_files = sorted(
        #         [os.path.join(checkpoint_dir, f)
        #          for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        #         key=os.path.getmtime
        #     )
        #     if checkpoint_files:
        #         latest_checkpoint = checkpoint_files[-1]
        #         print(
        #             f"Resuming training from checkpoint: {latest_checkpoint}")

        # # Initialize the Trainer
        # trainer = Trainer(
        #     devices=1,
        #     max_epochs=2000,
        #     logger=logger,
        #     callbacks=[checkpoint_callback],
        #     gradient_clip_val=1.0,
        # )

        # # Train the model, resuming from the latest checkpoint if it exists
        # trainer.fit(model, train_dataloaders=self.dataloader,
        #             val_dataloaders=self.dataloader, ckpt_path=latest_checkpoint)


if __name__ == '__main__':
    unittest.main()
