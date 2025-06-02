"""
Load a trained agent that can avoid pursuers and also allow capabilities 
to load the transformer model for deployment applications 

-This means I need to convert the inputs to feed it to my transformer model:
    - Get 20 past observations of the ego vehicle
    - Get 20 past observations of the pursuers
    - Batch the data and send into dataloader for processing


"""

import numpy as np
import unittest
import yaml
import torch
import matplotlib.pyplot as plt
import time 
import gc
import ray
import matplotlib.pyplot as plt
import pathlib
import torch
import numpy as np
import pickle as pkl

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from typing import List, Dict, Any
from typing import Dict, Any
from jarvis.utils.trainer import Trainer, load_yaml_config
from jarvis.envs.simple_multi_env import EngageEnv
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.envs.simple_agent import DataHandler, Pursuer, Evader
from jarvis.utils.vector import StateVector
from jarvis.envs.simple_agent import (
    SimpleAgent, PlaneKinematicModel, DataHandler,
    Evader, Pursuer)
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.mask import ActionMaskingRLModule
from jarvis.utils.trainer import RayTrainerSimpleEnv

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule

from jarvis.transformers.wayformer.dataset import LazyBaseDataset as BaseDataset
from jarvis.transformers.wayformer.predictformer import PredictFormer
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer



plt.close('all')
gc.collect()
# Used to clean up the Ray processes after training
ray.shutdown()
# For debugging purposes
# ray.init(local_mode=True)
ray.init()

def load_predictformer(
    model_config: str = "config/predictformer_config.yaml",
    data_config: str = "config/predictformer_config.yaml",
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 1
) -> PredictFormer:
    """
    Load a trained PredictFormer model and its dataset.
    
    Args:
        model_config (str): Path to the model configuration file.
        data_config (str): Path to the data configuration file.
        device (str): Device to load the model on ('cpu' or 'cuda').
        num_samples (int): Number of samples in the dataset.
        batch_size (int): Batch size for the DataLoader.
        
    Returns:
        PredictFormer: The loaded PredictFormer model.
    """
    
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset = BaseDataset(
        config=data_config,
        is_test=True,
        num_samples=num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    with open(model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    model = PredictFormer(
        config=model_config,
        device=device
    )
    return model, dataloader