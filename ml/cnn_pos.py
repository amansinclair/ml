import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_xy(h=32, w=32, n=500):
    X = np.zeros((n, 1, h, w), dtype="float32")
    y = np.zeros((n, 2), dtype="int64")
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


dataset = DataLoader(Grid(X, y), batch_size=128, shuffle=True)


def show_preds(net, ds):
    X, y = next(iter(ds))
    rows, cols = net(X)
    row_idx = torch.argmax(rows, axis=1)
    col_idx = torch.argmax(cols, axis=1)
    for i in range(len(y)):
        print(row_idx[i], col_idx[i], y[i])


def calc_exact(net, ds):
    X, y = next(iter(ds))
    rows, cols = net(X)
    row_idx = torch.argmax(rows, axis=1)
    col_idx = torch.argmax(cols, axis=1)
    n_exact = 0
    size = len(y)
    for i in range(size):
        if row_idx[i] == y[i, 0] and col_idx[i] == y[i, 1]:
            n_exact += 1
    print(n_exact / size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 9, 3)
        self.conv2 = nn.Conv2d(9, 18, 3)
        self.conv3 = nn.Conv2d(18, 36, 3)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(9)
        self.bn2 = nn.BatchNorm2d(18)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc_row = nn.Linear(144, 32)
        self.fc_col = nn.Linear(144, 32)

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.conv1(x))))
        x = self.bn2(self.pool(self.relu(self.conv2(x))))
        x = self.flatten(self.pool(self.relu(self.conv3(x))))
        row = self.fc_row(x)
        col = self.fc_col(x)
        return row, col


net = Net()
crit = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.01)

n_epochs = 50
for epoch in range(1, n_epochs + 1):
    losses = []
    for x, y in dataset:
        opt.zero_grad()
        rows, cols = net(x)
        loss = crit(rows, y[:, 0])
        loss += crit(cols, y[:, 1])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("epoch", epoch, torch.tensor(losses).mean())


show_preds(net, dataset)
calc_exact(net, dataset)

# TO DO - check eval with valid set
# min size of CNN
