import torch 
import torch.nn as nn

def create_jacobian_nn(self, n_joints, n_states, n_hidden_layers, activation = nn.ReLU, initializer = nn.init.xavier_uniform_):
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
    
    # Loss and optimizer
    # TODO: pass as parameters
    self._loss_f_jacobian = nn.MSELoss(reduction='sum')
    self._optim_jacobian = torch.optim.SGD(self.jacobian.parameters(), lr=1e-4)
    return

def forward_jacobian(self, x):
    J = self.jacobian(x)
    return J 

def loss_jacobian(self, x, y):
    n_samples = len(x)
    
    # reshape jacobians
    _j = self.forward_jacobian(x) 
    J = _j.reshape(n_samples, self.n_states, self.n_joints)
    
    # compute cartesian position of joints
    joints_cartesian = torch.empty(size=(n_samples, self.n_joints, self.n_states), requires_grad=False)
    for s in range(n_samples):
        for j in range(self.n_joints):
            joints_cartesian[s,:,j] = J[s, :, 0:j+1].matmul(x[s,0:j+1]) 
    
    # reshape labels
    _y = y.reshape(n_samples, self.n_states, self.n_joints)
    
    # accumulate cartesian error over all joints and all samples
    acc = 0
    for s in range(n_samples):
        for j in range(self.n_joints):
            acc += torch.sqrt(((joints_cartesian[s,:,j]-_y[s,:,j])**2).sum())

    return acc

def train_jacobian(self, x, y):
    self.jacobian.zero_grad()
    y_pred = self.jacobian(x)
    loss = self.jacobian_loss_f(y, y_pred)
    loss.backward()
    self.jacobian_optim.step()
