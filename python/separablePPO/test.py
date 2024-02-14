import torch
import torch.nn as nn
import itertools as it
import numpy as np
from matplotlib import pyplot as plt

class NN(torch.nn.Module):
    
    def __init__(self):
        self.nin = 2
        self.n1 = 5
        self.nglue = 4
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

    def forward(self, x):
        _x = self.nn1(x) + self.glue
        _y = self.nn2(_x)
        return _y

def f(x):
    r = (x[:,0]**2 + x[:,1]**2)
    return r.reshape(len(r), 1)

if __name__ == "__main__":
    nn = NN()
    optim = torch.optim.SGD(nn.parameters(), lr=1e-5)
    loss_f = torch.nn.MSELoss(reduction='sum')
    d_size = 500
    for i in range(10000):
        optim.zero_grad()
        
        x = torch.rand(d_size, 2)
        y = f(x)
        y_pred = nn(x)
        loss = loss_f(y, y_pred)
        print(loss)
        loss.sum().backward()
        optim.step()

    # test
    d_size = 100
    x = torch.rand(d_size, 2)
    y = f(x) 
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    with torch.no_grad():
        y_pred = nn(x).detach().numpy()
    #print(y, y_pred)
    ax.scatter(x[:,0], x[:,1], y, alpha = 0.5, label='label')
    ax.scatter(x[:,0], x[:,1], y_pred, alpha=0.5, label='pred')
    print(nn.glue)
    plt.legend()
    plt.show()


