import torch
import torch.nn as nn
import itertools as it
from matplotlib import pyplot as plt
import os

from NN import *

if __name__ == "__main__":
    nn = NN()
    
    # loss function and optmizer
    loss_f = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(nn.nn2.parameters(), lr=1e-4)
    
    loss_f_glue = torch.nn.MSELoss(reduction='sum')
    optim_glue = torch.optim.SGD(nn.nn1.parameters(), lr=1e-4)

    d_size = 500
    for i in range(5000):
        
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
    plots_folder = 'plots/'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plt.savefig('plots/test_separated.png')
    plt.show()


