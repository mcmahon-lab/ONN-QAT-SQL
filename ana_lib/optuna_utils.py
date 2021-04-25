import numpy as np
import os
from IPython.core.debugger import set_trace

import torch
import optuna
import pytorch_lightning as pl
import shutil

def create_study(NAS_name, sampler=optuna.samplers.TPESampler()):
    pruner = optuna.pruners.NopPruner()
    storage = f'sqlite:///{NAS_name}.db' #way to specify an SQL database
    study = optuna.create_study(pruner=pruner, sampler=sampler, 
            storage=storage, study_name="", load_if_exists=True)
    return study

def train_save(study, NAS_name, trainer, pl_model):
    checkpoint_cb = pl.callbacks.ModelCheckpoint()
    trainer.checkpoint_callback = checkpoint_cb
    
    if not isinstance(pl_model.configure_optimizers(), torch.optim.Optimizer):
        trainer.callbacks = [pl.callbacks.LearningRateLogger()]
    
    trainer.fit(pl_model)
    value = checkpoint_cb.best
    
    try:
        best_value = study.best_value
    except:
        best_value = np.inf
        
    if value < best_value:
        pl_model.load_from_checkpoint(checkpoint_cb.kth_best_model)
        torch.save(pl_model.model, f"{NAS_name}.p")
    
    return value

def setup_logger(trial, NAS_name, hparams):
    return pl.loggers.TensorBoardLogger("NAS", NAS_name, f"t{trial.number} {hparams}")

def SilentTrainer(*args, **kwargs):
    return pl.Trainer(*args, weights_summary=None, progress_bar_refresh_rate=0, **kwargs)

def delete_NAS(NAS_name):
    try:
        os.remove(f"{NAS_name}.p")
    except:
        print(f"{NAS_name}.p don't exist")
    try:
        os.remove(f"{NAS_name}.db")
    except:
        print(f"{NAS_name}.db don't exist")
    try:
        shutil.rmtree(f"NAS/{NAS_name}")
    except:
        print(f"NAS/{NAS_name} don't exist")