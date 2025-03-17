import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import unittest
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from jarvis.transformers.wayformer.dataset import LSTMDataset



class TestLSTMDataset(unittest.TestCase):
    
    def setUp(self):
        """
        pass
        """
        # Load configuration
        config_path = "config/lstm_config.yaml"  # adjust if needed
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create datasets for training and validation using your LazyBaseDataset
        self.train_dataset = LSTMDataset(config=config, is_validation=False)
        
    def test_get_data(self):
        """
        pass
        """
        print("Train dataset length:", len(self.train_dataset))
        self.assertTrue(len(self.train_dataset) > 0)
        
        # Create DataLoaders
        num_workers = 1
        batch_size = 1
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
        for i, batch in enumerate(train_dataloader):
            input_dict = batch['input_dict']
            if i == 1:
                break
            
        self.assertTrue(input_dict is not None)        
        obj_trajs:np.array = input_dict['obj_trajs']
        center_gt_trajs:np.array = input_dict['center_gt_trajs']
        print(obj_trajs.shape)
        print("center gt trajs", center_gt_trajs.shape)
        assert obj_trajs.shape[0] == batch_size
        assert center_gt_trajs.shape[0] == batch_size
        
        obj_trajs = obj_trajs.squeeze()
        center_gt_trajs = center_gt_trajs.squeeze()
        original_pos_past = input_dict['original_pos_past'].squeeze()
        center_world_trajs = input_dict['center_objects_world'].squeeze()
        
        fig, ax = plt.subplots()
        for i, obj_traj in enumerate(obj_trajs):
            ax.plot(obj_traj[:,0], obj_traj[:,1], label=f"Object {i}", linestyle='--', marker='o')
            zeroed_x = original_pos_past[i,:,0] - original_pos_past[i,0,0]
            zeroed_y = original_pos_past[i,:,1] - original_pos_past[i,0,1]
            ax.plot(zeroed_x, zeroed_y, label=f"Zeroed Pos {i}")
            #ax.plot(original_pos_past[i,:,0], original_pos_past[i,:,1], label=f"Original Pos {i}")
        
        ax.legend()
            
        fig, ax = plt.subplots()
        for i, center_gt_traj in enumerate(center_gt_trajs):
            ax.plot(center_gt_traj[:,0], center_gt_traj[:,1], label=f"Center {i}")
        ax.legend()
        
        fig, ax = plt.subplots()
        for i, center_world_traj in enumerate(center_world_trajs):
            ax.plot(center_world_traj[:,0], center_world_traj[:,1], label=f"Center World {i}")
        ax.legend()
        # plt.show()


if __name__ == '__main__':
    unittest.main()