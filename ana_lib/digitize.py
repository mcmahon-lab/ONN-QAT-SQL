import torch
import numpy as np

def digitizex(data, xlevels, fine_factor=20):
    #First do downsampling in x
    x=np.linspace(0,len(data),fine_factor*xlevels)
    datafine=np.interp(x,np.arange(0,len(data)),data)
    #now here get a new set of data that is mean over each thing
    datad = np.zeros(xlevels)
    for i in range(xlevels):
        datad[i] = np.sum(datafine[i*fine_factor:(i+1)*fine_factor])/fine_factor
    return datad

def digitizey(x, levels, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()
    x = (levels-1)*(x - min_val)/(max_val-min_val)
    x = x.round()
    x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)*(levels-1))
    x = x*(max_val-min_val)/(levels-1) + min_val
    return x

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels=2**4, min_val=None, max_val=None):
        x = digitizey(x, levels=levels, min_val=min_val, max_val=max_val)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

pt_digitize = FakeQuantOp.apply #Define the function that can used outside