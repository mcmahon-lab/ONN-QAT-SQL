import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)

# The pytorch model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, Nunits):
        super().__init__()
        self.layers = []

        for Nunit in Nunits:
            self.layers.append(nn.Linear(input_dim, Nunit))
            input_dim = Nunit
        
        self.layers.append(nn.Linear(input_dim, output_dim))

        # Assigning the layers as class variables (PyTorch requirement). 
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)
            
    def forward(self, data):
        for layer in self.layers[:-1]:
            data = layer(data)
            data = swish(data)
        data = self.layers[-1](data)
        return data