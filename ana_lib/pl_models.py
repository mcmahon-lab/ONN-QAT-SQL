import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class RegressionModel(pl.LightningModule):
    """
    Employs Mean Square Error loss to perform regression
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        return result