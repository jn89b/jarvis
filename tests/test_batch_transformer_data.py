import numpy as np
import unittest
import yaml
import torch

from jarvis.transformers.wayformer.dataset import BaseDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(unittest.TestCase):

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

        for i, batch in enumerate(self.dataloader):
            print(batch)
            # make sure the input_dictionary values are torch tensors
            input_dict = batch['input_dict']
            for k, v in input_dict.items():
                print(k, v)
            # self.assertTrue(isinstance(batch['input_dict'], torch.Tensor))


if __name__ == '__main__':
    unittest.main()
