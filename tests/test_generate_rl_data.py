
from jarvis.envs.env import DynamicThreatAvoidance, EnvConfig, AircraftConfig
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.agent import Agent
from aircraftsim import SimInterface
from aircraftsim.utils.report_diagrams import SimResults
from jarvis.utils.trainer import Trainer, load_yaml_config

from gymnasium import space
from typing import List
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import unittest
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('TkAgg')
USE_IMPORT = True


class TestGeneratedData(unittest.TestCase):

    def setUp(self):
        config_dir: str = "config/training_config.yaml"
        # Load the YAML file
        config = load_yaml_config(config_dir)

        # Access each configuration component
        model_config = config.get('model_config', {})
        env_config = config.get('env_config', {})
        training_config = config.get('training_config', {})
        self.trainer = Trainer(model_config=model_config,
                               env_config=env_config,
                               training_config=training_config)

    def test_generate_dataset(self):
        self.trainer.generate_dataset()
        data_dir = self.trainer.training_config['data_dir']


if __name__ == '__main__':
    unittest.main()
