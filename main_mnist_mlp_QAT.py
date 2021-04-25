"""
This training script is a self-contained minimalist version of quantization aware training (QAT)
for multi-layer perceptrons (MLP) on the MNIST dataset. 
It only requires pytorch (1.7.0) and torchvision (0.8.1) packages to run. 

To obtain the best models, batches of models should be trained with randomly sampled training parameters,
which are included together with training in ./mnist_mlp_QAT_batch_training.ipynb. 
The models should then be selected based on their resilience to photon shot noise, 
which is simulated and evaluated in ./model_evaluation_shot_noise_sim.ipynb 
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_decay', type=float, default=0.5, metavar='LRD',
                    help='learning rate decay factor')
parser.add_argument('--momentum', type=float, default=0.87, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--warmup_eps', type=int, default=6, metavar='WRM',
                    help='number of initial epoches without activation digitization')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=0,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data Augmentation Transforms
def conv1d(NCHW_tensor):
    conv_ker = torch.tensor([[0.05, 0.1, 0.05], [0.1, 1, 0.1], [0.05, 0.1, 0.05]])
    conv_ker = conv_ker.view(1,1, conv_ker.size(0), conv_ker.size(1))
    return F.conv1d(NCHW_tensor.unsqueeze(0), conv_ker, padding=1).squeeze(0)

transforms_distort = transforms.Compose([transforms.RandomAffine(5, translate=(0.04, 0.04), scale=(0.96, 1.04)),
                                       transforms.ToTensor(),
                                       transforms.Lambda(conv1d)])

# Prepare data loaders
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./ML_data', train=True, download=True,
                   transform=transforms_distort),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./ML_data', train=False, 
                   transform=transforms_distort),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Define digitization operation and digitized linear layers.
def Digitize(tensor, quant_mode='det', levels=16, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = tensor.min(), tensor.max()
    tensor.clamp_(min_val, max_val).add_(-1*min_val).mul_(levels-1).div_(max_val-min_val)
    if quant_mode == "det": 
        tensor.round_()
    elif quant_mode == "rand":
        tensor.add_(torch.rand(tensor.size(), device=tensor.device).add_(-0.5)).round_()
    tensor.mul_(max_val-min_val).div_(levels-1).add_(min_val)
    return tensor

class DigitizeLinear(nn.Linear):

    def __init__(self,  *kargs, a_quant_mode="det", w_quant_mode="det", a_quant_levels=16, w_quant_levels=32, running_weight=0.001, **kwargs):
        super(DigitizeLinear, self).__init__(*kargs, **kwargs)
        self.act_quant_mode = a_quant_mode
        self.weight_quant_mode = w_quant_mode
        self.register_buffer("act_quant_levels", torch.tensor(a_quant_levels))
        self.register_buffer("weight_quant_levels", torch.tensor(w_quant_levels))
        self.register_buffer("running_weight", torch.tensor(running_weight)) 
        self.register_buffer("running_min", None)
        self.register_buffer("running_max", None)

    def forward(self, input):

        if not self.weight_quant_mode is None: # Set a flag to control weight digitization.
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.data=Digitize(self.weight.data, quant_mode=self.weight_quant_mode, levels=self.weight_quant_levels)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()    
        out = nn.functional.linear(input, self.weight, bias=self.bias)

        if not self.act_quant_mode is None: # A flag to control output digitization. 
            if self.training: # Update the running average of min and max only during training
                with torch.no_grad():
                    if not self.running_min and not self.running_max:
                        self.running_min, self.running_max = out.min(), out.max()
                    self.running_min = (1-self.running_weight) * self.running_min + self.running_weight * out.min()
                    self.running_max = (1-self.running_weight) * self.running_max + self.running_weight * out.max()
            out.data=Digitize(out.data, quant_mode=self.act_quant_mode, levels=self.act_quant_levels, min_val=self.running_min, max_val=self.running_max)
    
        return out

# Define the class for generic MLPs with digitized weights and activations.
class Net(nn.Module):
    def __init__(self, Nunits, **kwargs):
        super().__init__()
        self.fcs = nn.ModuleList([DigitizeLinear(i,j,**kwargs) for i, j in zip(Nunits[:-1], Nunits[1:])])

    def forward(self, X):
        X = X.view(X.size(0), -1)
        for i, fc in enumerate(self.fcs):
            X = fc(X)
            if fc is not self.fcs[-1]:
                X = F.relu(X)
        return X
    
    def set_digitize_config(self, a_quant_mode, w_quant_mode, a_quant_levels, w_quant_levels):
        for fc in self.fcs:
            fc.act_quant_mode = a_quant_mode
            fc.weight_quant_mode = w_quant_mode
            fc.act_quant_levels = torch.tensor(a_quant_levels)
            fc.weight_quant_levels = torch.tensor(w_quant_levels)
        
# Specify model size and digitization configurations, and then construct the MLP model.
Nunits = [784, 100, 100, 10]
model = Net(Nunits, a_quant_mode=None, w_quant_mode=None, a_quant_levels=16, w_quant_levels=32)
if args.cuda:
    torch.cuda.set_device(0) # Set the GPU index
    model.cuda()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Define the train loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
     
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define the test loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))

# Train the model
for epoch in range(1, args.epochs + 1):
    if epoch > args.warmup_eps:
        model.set_digitize_config("rand", "det", 16, 32)
    train(epoch)
    accu = test()
    if epoch%20==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*args.lr_decay

# Save the trained model
modelSavePath = "./debug"
if not os.path.exists(modelSavePath):
    os.makedirs(modelSavePath)
torch.save(model.state_dict(), modelSavePath + "/test.pt")
