# Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os

import torch

from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# our definitions
from NN import NN

verbose = False
model_folder = 'models/'
plots_folder = 'plots/'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

is_fork = multiprocessing.get_start_method() == "fork"
print('is fork: ', is_fork)
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
print('Device: ', device)

lr = 3e-4
frames_per_batch = 1000

# Enviroments
base_env = gym.make("Reacher-v4", render_mode='human')

observation_size = 2
action_size = 2
n_joints = 2
n_states = 2

print('obs, action sizes: ', observation_size, action_size)
# Define the Network
net = NN()
net.create_jacobian_nn(n_joints, n_states, [10, 10, 10])

optim = torch.optim.Adam(net.parameters(), lr=lr)

for _ in range(num_epochs):
    obs, info = env.reset()
    for _ in range(frames_per_batch):
        action = env.rand_action()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        _sc = obs[0:4].reshape(2, n_joints)
        print(_sc)
        angles = torch.atan2(_sc[1,:], _sc[0,:])
        print(angles)
        # TODO: get cart. position of each joint
        y = 
        optim = zero_grad()
        loss = net.loss_jacobian(angles, y)
        print('loss: ', loss)
        loss.backward()
        optim.step()

