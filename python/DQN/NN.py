import torch
import torch.nn as nn
import itertools as it
import numpy as np

class NN(torch.nn.Module):
    
    def __init__(self, device, dimentions):
        super(NN, self).__init__()
        self.device = device
        self.dimentions = dimentions

        self.model = nn.Sequential()
        self.model.append(nn.Linear(dimentions[0], dimentions[1])) #input

        self.compute_action_list(2, -0.5, 0.5, 0.25)
        
        for n_in, n_out in zip(dimentions[1:-2], dimentions[2:-1]):
            self.model.append(nn.Linear(n_in, n_out))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(dimentions[-2], len(self.action_list))) #output
        #self.model.append(nn.Softmax())

        # initialize the NN
        for l in self.model.children():
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_uniform_(l.weight)
                l.bias.data.fill_(0.0)

    def forward(self, x):
        return self.model(x)

    def compute_action_list(self, dim, low, high, step):
        n = int(np.ceil((high-low)/step))
        _list_1d = np.linspace(low, high, n)
        self.action_list = list(it.combinations_with_replacement(_list_1d, dim))










