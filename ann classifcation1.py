#modified from the original :Mike X Cohen course "A deep understanding of deep learning"
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
#display.set_matplotlib_formats('svg')

def data(nPerClust, blur, A , B, plot=False):
    a = [A[0] + np.random.randn(nPerClust)*blur, A[1] + np.random.randn(nPerClust)*blur]
    b = [B[0] + np.random.randn(nPerClust)*blur, B[1] + np.random.randn(nPerClust)*blur]
    labels_np = np.vstack((np.zeros((nPerClust, 1)), np.ones((nPerClust, 1))))
    data_np = np.hstack((a,b)).T
    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()
    if plot:
        plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], 'bs')
        plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], 'ko')
        plt.title("The qwerties")
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.show()
    return data, labels


def model():
    # model
    ANNclassify = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1),
        #nn.Sigmoid()
    )
    return ANNclassify

def createANNModel(learningRate):
    model_ann = model()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model_ann.parameters(), lr=learningRate)
    return model_ann, loss_func, optimizer

def train_model(model_ann, data, labels, loss_func, optimizer, epochs):
    losses = torch.zeros(epochs)
    for epoch in range(epochs):
        output = model_ann(data)

        loss = loss_func(output, labels)
        losses[epoch] = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    pred = model_ann(d)
    totalacc = 100*torch.mean(((pred>0)==labels).float())
    return losses, pred, totalacc


A = [1, 1]
B = [5, 1]
nPerClust = 100
d, l = data(nPerClust, 1, A, B)
learningRate = 0.07
numEpochs = 1000
ANNclassify, lossesFunc, optimizer = createANNModel(learningRate)
losses, prediction, totalacc = train_model(ANNclassify, d, l, lossesFunc, optimizer, numEpochs)

plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=0.1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("final accuracy: %g%%" %totalacc)

if False:
    fig = plt.figure(figsize=(5, 5))
    plt.plot(d[miss, 0], d[miss, 1], 'rx', markersize=12)
    plt.plot(d[np.where(~predlabel)[0], 0], d[np.where(~predlabel)[0], 1], 'bs')
    plt.plot(d[np.where(predlabel)[0], 0], d[np.where(predlabel)[0], 1], 'ko')
    plt.title("The qwerties")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(['Miss', 'blue', 'black'], bbox_to_anchor=(1, 1))

    plt.show()

learningRates = np.linspace(0.001, 0.2, 100)
accByLR = []
allLosses = np.zeros((len(learningRates), numEpochs))

for i, lr in enumerate(learningRates):
    ANNclassify, lossesFunc, optimizer = createANNModel(lr)
    losses, prediction, totalacc = train_model(ANNclassify, d, l, lossesFunc, optimizer, numEpochs)

    accByLR.append(totalacc)
    allLosses[i, :] = losses.detach()

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(learningRates, accByLR, 's-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title("Accuracy by Learning rate")

ax[1].plot(allLosses.T)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title("Loss by Epoch")

plt.show()


