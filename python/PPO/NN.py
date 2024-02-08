import torch
import torch.nn as nn
import itertools as it
import numpy as np

# Torch imports
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

class NN(torch.nn.Module):
    
    def __init__(self, env):
        super(NN, self).__init__()
        self.device = env.device 
        self.env = env

    def define_actor_network(self, dimentions):
        self.actor_dimentions = dimentions
        self.actor = nn.Sequential()
 
        for n in dimentions[:-1]:
            self.actor.append(nn.LazyLinear(n))
            self.actor.append(nn.Tanh())
        self.actor.append(nn.LazyLinear(dimentions[-1])) #output
        self.actor.append(NormalParamExtractor())
       
        _policy_module = TensorDictModule(
            self.actor, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        self.policy_module  = ProbabilisticActor(
                module = _policy_module,
                spec = self.env.action_spec,
                in_keys = ["loc", "scale"],
                distribution_class = TanhNormal,
                distribution_kwargs = {
                    "min": self.env.action_spec.space.low,
                    "max": self.env.action_spec.space.high,
                },
                return_log_prob=True,
                )
        return

    def define_value_network(self, dimentions):
        self.value_dimentions = dimentions
        self.value_net = nn.Sequential()

        for n in dimentions[:-1]:
            self.value_net.append(nn.LazyLinear(n))
            self.value_net.append(nn.Tanh())
        self.value_net.append(nn.LazyLinear(dimentions[-1]))
        
        self.value_module = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
        )
        return

    def forward(self, x):
        return self.actor(x)
