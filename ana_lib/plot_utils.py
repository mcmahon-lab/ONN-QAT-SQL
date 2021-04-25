import matplotlib.pyplot as plt
import numpy as np
import torch

def meshgrid_center(xdata, ydata):
    x = (xdata[:-1]+xdata[1:])/2
    x = np.array([2*xdata[0]-x[0], *x, 2*xdata[-1]-x[-1]])
    y = (ydata[:-1]+ydata[1:])/2
    y = np.array([2*ydata[0]-y[0], *y, 2*ydata[-1]-y[-1]])
    return x, y

def pcolormesh_center(x, y, Z, **kws):
    X, Y = meshgrid_center(x, y)
    return plt.pcolormesh(X, Y, Z.T, **kws)

def plot_max(x, y, *args, **kwargs):
    plt.plot(x, y/np.max(y), *args, **kwargs)

def plot_grid(y_pred, y, ylim=None):
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for (ind, ax) in enumerate(np.ndarray.flatten(axs)):
        ind = ind 
        plt.sca(ax)
        plt.plot(y_pred[ind], "k.--", label="pred")
        plt.plot(y[ind], ".--", alpha=0.6, label="test")
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
    plt.legend()
    return fig, axs

def plot_equal(y, y_pred, *args, **kwargs):
    ymin = np.minimum(y.min(), y_pred.min())
    ymax = np.maximum(y.max(), y_pred.max())
    plt.plot([ymin, ymax], [ymin, ymax], *args, **kwargs)

def plot_equal_cuda(y, y_pred, *args, **kwargs):
    plot_equal(y.detach().cpu(), y_pred.detach().cpu(), *args, **kwargs)

def plot_cuda(*args, **kwargs):
    if len(args) == 1:
        plt.plot(args[0].detach().cpu(), *args[1:], **kwargs)
    else:
        if not isinstance(args[1], torch.Tensor):
            plt.plot(args[0].detach().cpu(), *args[1:], **kwargs)
        else:
            plt.plot(args[0].detach().cpu(), args[1].detach().cpu(), *args[2:], **kwargs)


def make_f(param_f1, param_f2):
    f1 = make_f1(**param_f1)
    f2 = make_f2(**param_f2)
    return f1, f2

