"""
Simple functions for performing basic data wrangling of the spectrums
"""
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

from .digitize import *

def get_loader(xlist, specs_list, process_spec, Nbatch=200):
    spectrums = np.array([[process_spec(s) for s in specs] for specs in specs_list])
    spectrums = spectrums.swapaxes(0, 1)

    Nrepeat = spectrums.shape[1]
    Nlam = spectrums.shape[2]

    Ntotal = spectrums.shape[0]
    np.random.seed(0)
    perm_xlist = np.random.permutation(Ntotal)
    xlist = xlist[perm_xlist, :]
    spectrums = spectrums[perm_xlist, :, :]

    mean_spectrums = np.mean(spectrums, axis=1)
    std_spectrums = np.std(spectrums, axis=1)

    # Create the dataset
    Ntotal = mean_spectrums.shape[0]

    train_ratio = 0.90

    Ntrain = int(np.floor(Ntotal*train_ratio))
    train_inds = np.arange(Ntrain)
    val_inds = np.arange(Ntrain, Ntotal)

    X_train = torch.tensor(xlist[train_inds]).float()
    X_train = torch.stack([X_train for i in range(Nrepeat)], axis=1)
    X_train = X_train.reshape((-1, X_train.shape[2]))

    X_val = torch.tensor(xlist[val_inds]).float()
    X_val = torch.stack([X_val for i in range(Nrepeat)], axis=1)
    X_val = X_val.reshape((-1, X_val.shape[2]))

    Y_train = torch.tensor(spectrums[train_inds, :, :]).float()
    Y_train = Y_train.reshape((-1, Y_train.shape[2]))

    Y_val = torch.tensor(spectrums[val_inds, :, :]).float()
    Y_val = Y_val.reshape((-1, Y_val.shape[2]))

    torch.manual_seed(0)
    perm = torch.randperm(Y_train.shape[0])
    X_train = X_train[perm, :]
    Y_train = Y_train[perm, :]

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, Nbatch)
    val_loader = DataLoader(val_dataset, val_dataset.tensors[0].shape[0])
    return train_loader, val_loader