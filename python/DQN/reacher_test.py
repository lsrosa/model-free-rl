import gym, torch
import os, glob, re

# Our stuff
from NN import NN
from common import * 
import time 

model = NN(device, dimentions).to(device)

# get the model with most episodes on the folder
model_files = glob.glob('models/model*.pt')
episodes = [int(re.findall(r'\d+', file_name)[0]) for file_name in model_files]
n = max(episodes)

# load model
print('loading model %d'%n)
model.load_state_dict(torch.load('models/model%d.pt'%n))
model.eval()

# GYM defs
env = gym.make('Reacher-v4', render_mode='human')
n_epi = 10
max_steps = 100

for epi in range(n_epi): #episodes
    print('episode: ', epi)
    observation, info = env.reset()
    acc_reward = 0

    for step in range(max_steps):
        env.render()

        _action = policy(observation, model).data.max(1)[1].view(1, 1)
# User-defined policy function
        observation, _reward, terminated, truncated, info = env.step(action_interpreter(_action, model))

        # accumulate all rewards from this episode
        acc_reward += _reward
        
        if terminated or truncated:
            print('terminated')
            break

        time.sleep(0.1)
env.close()
