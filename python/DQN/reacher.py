## Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
## DQN learning

import gym, torch
import numpy as np
import random, math
import os

# Our stuff
from NN import NN
from ReplayBuffer import ReplayBuffer, Transition
from common import * 

model_t = NN(device, dimentions).to(device)
model_p = NN(device, dimentions).to(device)
model_t.load_state_dict(model_p.state_dict())

# Training hyperparams
n_epi = 10000
max_steps = 500
avg_steps = []
batch_size = 128 
discount_factor = 0.99
target_update = 10
learning_rate = 1e-4
tau = 0.005 # softmax weght

# define optmizer
optmizer = torch.optim.AdamW(model_p.parameters(), lr=learning_rate, amsgrad=True)

def train(batch_size, discount_factor):

    if len(buffer) < batch_size:
        return

    transitions = buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.uint8).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(torch.int64).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model_p(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size).to(device)
    with torch.no_grad():
        next_state_values[non_final_mask] = model_t(non_final_next_states).max(1).values
    
    #print(next_state_values)   
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * discount_factor) + reward_batch
    #print('\n\n expected: ', expected_state_action_values)
    #print('state action: ', state_action_values)
    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss() 
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optmizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model_p.parameters(), 100)
    optmizer.step()
    return

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done/EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            _result = policy(state, model_p)#[0]
       
        _a = _result.data.max(1)[1].view(1, 1)
        return _a
    else:
        _a = Tensor([[random.randrange(len(model_t.action_list))]])
        return _a

def save_model(path, model_number, rewards):
    if not os.path.exists(path):
        os.makedirs(path)
    
    model_file = path + '/model%d.pt'%model_number
    print(model_file)
    torch.save(model_t.state_dict(), model_file)
    np.save(path+'/rewards%d'%model_number, rewards)
    return

# Gym session
env = gym.make('Reacher-v4')
#env = gym.make('Reacher-v4', render_mode='human')

# Save data for training
buffer = ReplayBuffer(10000)

# Run the learning loop
epi_rewards = []
for epi in range(n_epi): #episodes
    print('episode: ', epi)
    observation, info = env.reset()
    cur_state = get_state(observation)
    acc_reward = 0
    #env.render()
    for step in range(max_steps):
        #env.render()

        _action = select_action(observation) # User-defined policy function
        observation, _reward, terminated, truncated, info = env.step(action_interpreter(_action, model_t))
        _reward = abs(_reward) # this env returns a negativa value of loss
        #print('reward info: ', info)
        #print('state, action, reward: ', cur_state.numpy(), _action, _reward)

        # accumulate all rewards from this episode
        acc_reward += _reward
        if terminated:
            next_state = None
            avg_steps.append(step)
        else:
            next_state = get_state(observation)
       
        # Tensor everything before sending to buffer
        action = _action
        reward = Tensor([_reward])
        buffer.push(cur_state, action, next_state, reward)
        
        cur_state = next_state
        train(batch_size, discount_factor)
    
        # Softmax Update the target network
        #if np.mod(step, target_update) == 0: 
        cp_targetParam = model_t.state_dict()
        cp_policyParam = model_p.state_dict()

        for k in cp_targetParam:
            cp_targetParam[k] = tau*cp_policyParam[k] + (1-tau)*cp_targetParam[k]
        model_t.load_state_dict(cp_targetParam)
        
        if terminated or truncated: break

    epi_rewards.append(acc_reward)
    if np.mod(epi, 100) == 0 and epi > 1:
        save_model(cur_path+'/models', epi, np.array(epi_rewards))
env.close()
