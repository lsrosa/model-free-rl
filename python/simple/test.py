import numpy as np
import torch

loss_f = torch.nn.MSELoss(reduction='sum')
x = torch.rand((3, 7))
y = torch.ones((3, 2*7))

def my_loss(y, y_pred):
    print(y, y_pred)
    a = y.reshape((len(y), 2, 7))
    b = y_pred.reshape((len(y_pred), 2, 7))
    print(y, y_pred)

    acc = 0
    for s in range(3):
        for j in range(7):
            acc += torch.sqrt(((a[s, :, j]-b[s,:,j])**2).sum())
    return acc

model = torch.nn.Linear(7, 2*7)
print('before', model.parameters())
y_pred = model(x)
optim = torch.optim.SGD(model.parameters(), lr=1e-4)
loss = my_loss(y, y_pred)
loss.backward()
print('after', model.parameters())

print(loss)
