import torch
import torch.nn as nn
import itertools as it
import numpy as np
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)

class NN(torch.nn.Module):
    
    def __init__(self):
        self.nin = 2
        self.n1 = 5
        self.nglue = 1
        self.n2 = 10
        self.nout = 1

        super(NN, self).__init__()
        self.nn1 = nn.Sequential()
        self.nn1.append(nn.Linear(self.nin, self.n1))
        self.nn1.append(nn.ReLU())
        self.nn1.append(nn.Linear(self.n1, self.n1))
        self.nn1.append(nn.ReLU())
        self.nn1.append(nn.Linear(self.n1, self.n1))
        self.nn1.append(nn.ReLU())
        self.nn1.append(nn.Linear(self.n1, self.nglue))
        
        for l in self.nn1.children():
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_uniform_(l.weight)
                l.bias.data.fill_(0.0)

        self.nn2 = nn.Sequential()
        self.nn2.append(nn.Linear(self.nglue+1, self.n2))
        self.nn1.append(nn.ReLU())
        self.nn2.append(nn.Linear(self.n2, self.n2))
        self.nn1.append(nn.ReLU())
        self.nn2.append(nn.Linear(self.n2, self.n2))  
        self.nn1.append(nn.ReLU())
        self.nn2.append(nn.Linear(self.n2, self.nout))
        
        self.glue = torch.empty(self.nglue, requires_grad = True)
        
        for l in self.nn2.children():
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_uniform_(l.weight)
                l.bias.data.fill_(0.0)

    def forward_glue(self, x):
        return self.nn1(x)
    
    def forward_all(self, x):
        _x = self.nn1(x)
        x1 = x[:,1].reshape(len(x), 1) 
        _y = self.nn2(torch.concat((_x, x1), dim=1))
        return _y 

    def forward(self, x):
        _x = self.nn1(x) 
        x1 = x[:,1].reshape(len(x), 1) 
        _y = self.nn2(torch.concat((_x, x1), dim=1))
        return _y 

def f_glue(x):
    r = x[:, 0]**2
    return r.reshape(len(r), 1)

def f(x):
    r = (x[:,0]**2 + x[:,1]**2)
    return r.reshape(len(r), 1)

def freeze(nn, f = True):
    for p in nn.parameters():
        p.requires_grad_(not f)
    return

def print_params(nn):
    print('\n\nnn1: ')
    for p in nn.nn1.parameters():
        print(p)
    print('\n\nnn2: ')
    for p in nn.nn2.parameters():
        print(p)
    return
