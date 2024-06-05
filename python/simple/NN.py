import torch
import torch.nn as nn
import itertools as it
import numpy as np

# Torch imports
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


class NN(torch.nn.Module):
    # Sub networks
    from jacobian import create_jacobian_nn, forward_jacobian, loss_jacobian 
    
    def __init__(self, activation_type = nn.ReLU, initializer = nn.init.xavier_uniform_):
        super(NN, self).__init__()
        self.activation_type = activation_type
        self.initializer = initializer

    def forward(self, x):
        return self.actor(x)
