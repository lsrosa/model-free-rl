import gym, torch
import os, glob, re
import multiprocessing
from matplotlib import pyplot as plt

# Our stuff
from NN import NN
import time 

from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

# Env defs
is_fork = multiprocessing.get_start_method() == "fork"
print('is fork: ', is_fork)
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
print('Device: ', device)

base_env = GymEnv("Reacher-v4", render_mode='human', device=device)

# Transforms the env for collecting some data
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

# collects data on the transformed env
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)

# create, define and load the model
num_cells = 256  # number of cells in each layer i.e. output dim.
model = NN(env)
model.define_actor_network([num_cells, num_cells, num_cells, 2*env.action_spec.shape[-1]])
# load model
model.actor.load_state_dict(torch.load('models/model.pt'))
model.actor.eval()
model.to(device)

rewards = []
for i in range(10):
    ro = env.rollout(100, model.policy_module)
    rewards.append(ro['next', 'reward'].cpu().mean().item())

plt.figure()
plt.plot(rewards)
plt.show()
