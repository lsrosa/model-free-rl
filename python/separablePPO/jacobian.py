import torch 
import torch.nn as nn

def create_jacobian_nn(self, n_joints, n_states, n_hidden_layers, activation = nn.ReLU, initializer = nn.init.xavie_uniform_):
    # save these for later use
    self.n_joints = n_joints
    self.n_states = n_states
    self.h_dim = n_hidden_layers

    self.jacobian = nn.Sequential()
    
    self.jacobian.append(nn.Linear(n_joints, n_hidden_layers[0]))
    self.jacobian.append(activation())
    for n_in, n_out in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
        self.jacobian.append(nn.Linear(n_in, n_out))
        self.jacobian.append(activation())
    self.jacobian.append(nn.Linear(n_hidden_layers[-1], n_states*n_joints))

    for l in self.jacobian.children():
        if isinstance(l, nn.Linear):
            initializer(l.weight)
            l.bias.data.fill_(0.0)
    return

    self.joint_cartesian_states = torch.empty(size=(n_joints, n_states), requires_grad=True)
    return

def forward_jacobian(self, x):
    # be mindful of this order
    J = self.jabobian(x).reshape(self.n_states, self.n_joints)

    # this should be in the loss only for now
    for i in range(len(J)):
        self.joint_cartesian_states[i] = J[:, 0:i+1].matmul(x[0:i+1]) 

    return self.jacobian(x)

def train_jacobian(self, x, y):
    self.jacobian.zero_grad()
    y_pred = self.jacobian(x)
    loss = self.jacobian_loss_f(y, y_pred)
    loss.backward()
    self.jacobian_optim.step()
