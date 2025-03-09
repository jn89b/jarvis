import json

import numpy as np
import pytorch_lightning as pl
import torch

class BaseModelV2(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pred_dicts = []

    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, future_len, 2)
                }
                loss (with gradient)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        # self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def on_validation_epoch_end(self):
    #     # if self.config.get('eval_waymo', False):
    #     #     metric_results, result_format_str = self.compute_metrics_waymo(
    #     #         self.pred_dicts)
    #     #     print(metric_results)
    #     #     print(result_format_str)

    #     # elif self.config.get('eval_nuscenes', False):
    #     #     import os
    #     #     os.makedirs('submission', exist_ok=True)
    #     #     json.dump(self.pred_dicts, open(os.path.join(
    #     #         'submission', "evalai_submission.json"), "w"))
    #     #     metric_results = self.compute_metrics_nuscenes(self.pred_dicts)
    #     #     print('\n', metric_results)

    #     # elif self.config.get('eval_argoverse2', False):
    #     #     metric_results = self.compute_metrics_av2(self.pred_dicts)

    #     self.pred_dicts = []

    def configure_optimizers(self):
        raise NotImplementedError

        return
