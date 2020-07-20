import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_xy(h=32, w=32, n=500):
    X = np.zeros((n, 1, h, w), dtype="float32")
    y = np.zeros((n, 2), dtype="float32")
    for i in range(n):
        row = np.random.choice(32)
        col = np.random.choice(32)
        X[i, 0, row, col] = 1.0
        y[i] = [row, col]
    return torch.from_numpy(X), torch.from_numpy(y)


X, y = get_xy()


class Grid(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = DataLoader(Grid(X, y), batch_size=32, shuffle=True)


def show_preds(net, ds):
    X, y = next(iter(ds))
    preds = net(X)
    for i in range(10):
        print(preds[i], y[i])


net = nn.Sequential(
    nn.Conv2d(1, 9, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.BatchNorm2d(9),
    nn.Conv2d(9, 18, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.BatchNorm2d(18),
    nn.Conv2d(18, 36, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(144, 2),
)
crit = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=0.01)

n_epochs = 200
for epoch in range(1, n_epochs + 1):
    losses = []
    for x, y in dataset:
        opt.zero_grad()
        preds = net(x)
        loss = crit(preds, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("epoch", epoch, torch.tensor(losses).mean())

show_preds(net, dataset)
