import os
import torch
import numpy as np

# Env definitions
cur_path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.Tensor if device=='cuda' else torch.Tensor

# NN defs
dimentions = [8, 128, 128, 2] # 5 inputs, 4x10 hidden, 2 output

def get_state(observation):
    return Tensor(observation[0:8].reshape(1, 8))

# from action index to joint velocity
def action_interpreter(action_index, model):
    return np.array(model.action_list[int(action_index[0])])

def policy(observation, model):
    x = get_state(observation)    
    x = x.to(device)
    y = model(x)
    
    # Send it back to cpu
    y = y.to('cpu')
    return y
