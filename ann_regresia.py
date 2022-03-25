#modified from the original :Mike X Cohen course "A deep understanding of deep learning"


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
#display.set_matplotlib_formats('svg')

def data(m):
    # data
    N = 50
    x = torch.randn(N, 1)
    y = m * x + torch.randn(N, 1) / 2
    return x,y

def model(m, x, y):
    # model
    ANN_reg_model = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )

    learning_rate = 0.1

    loss_function = nn.MSELoss()

    optimizer = torch.optim.SGD(ANN_reg_model.parameters(), lr=learning_rate)

    num_epoches = 300
    loss_array = torch.zeros(num_epoches)

    # train
    for epoch in range(num_epoches):
        output = ANN_reg_model(x)

        loss = loss_function(output, y)
        loss_array[epoch] = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred = ANN_reg_model(x)
    final_loss = (pred - y).pow(2).mean()

    """plt.plot(loss_array.detach(), range(num_epoches), '-o', )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('final loss = {}'.format(round(final_loss.item(), 3)))
    plt.show()"""
    return final_loss.item()

slopes = np.linspace(-2, 2, 21)
losses = []
for slope in slopes:
    loss = 0
    n = 10
    x,y=data(slope)
    for i in range(n):
        loss += model(slope,x,y)
    print(slope, loss/n)

    losses.append(loss/n)


plt.plot(slopes, losses)
plt.xlabel("slopes")
plt.ylabel("loss")
plt.show()

"""plt.plot(x, y, 'bo', label='data')
plt.plot(x, pred.detach(), 'rs', label="predictions")
plt.legend()
plt.show()"""



