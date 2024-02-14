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
        self.nn2.append(nn.Linear(self.nglue, self.n2))
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
        x = self.nn1(x) 
        x = self.nn2(x)
        return x 

    def forward(self, x):
        x = self.nn1(x) 
        x = self.nn2(x)
        return x 

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

if __name__ == "__main__":
    nn = NN()
    
    # loss function and optmizer
    loss_f = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(nn.nn2.parameters(), lr=1e-5)
    
    loss_f_glue = torch.nn.MSELoss(reduction='sum')
    optim_glue = torch.optim.SGD(nn.nn1.parameters(), lr=1e-5)

    d_size = 500
    for i in range(10000):
        
        # labels
        x = torch.rand(d_size, 2)
        
        # train glue
        #print('!!!!!!before')
        #print_params(nn) 
        
        y_glue = f_glue(x)
        optim_glue.zero_grad()
        y_pred_glue = nn.forward_glue(x)
        loss_glue = loss_f_glue(y_glue, y_pred_glue)
        #freeze(nn.nn1, False)
        loss_glue.backward()
        optim_glue.step()
         
        #print('!!!!!!after1')
        #print_params(nn) 

        #freeze(nn.nn1, True)
        y = f(x)
        optim.zero_grad()
        y_pred = nn.forward_all(x)
        loss = loss_f(y, y_pred)
        loss.backward()
        optim.step()

        #print('!!!!!!after2')
        #print_params(nn) 
        print('loss glue, loss: ', loss_glue.data, loss.data)
    # test
    d_size = 100
    x = torch.rand(d_size, 2)
    y = f(x) 
    y_glue = f_glue(x)

    with torch.no_grad():
        y_pred_glue = nn.nn1(x).detach().numpy()
        y_pred = nn(x).detach().numpy()

    fig = plt.figure()

    #print(y, y_pred)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x[:,0], x[:,1], y, alpha = 0.5, label='label')
    ax.scatter(x[:,0], x[:,1], y_pred, alpha=0.5, label='pred')
    plt.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x[:,0], y_glue, alpha=0.5, label='label')
    ax.scatter(x[:,0], y_pred_glue, alpha=0.5, label='pred')
    plt.legend()

    plt.show()


