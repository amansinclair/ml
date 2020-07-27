import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import product


def get_xy(h=32, w=32):
    y = np.array(list(product([i for i in range(32)], repeat=2)), dtype="int64")
    np.random.shuffle(y)
    X = np.zeros((h * w, 1, h, w), dtype="float32")
    i = 0
    for row, col in y:
        X[i, 0, row, col] = 1.0
        i += 1
    return torch.from_numpy(X), torch.from_numpy(y)


X, y = get_xy()
train_size = 600
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]


class Grid(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.do1 = nn.Dropout2d(p=0.25)
        self.do2 = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc_row = nn.Linear(64, 32)
        self.fc_col = nn.Linear(64, 32)

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.do1(self.conv1(x)))))
        x = self.bn2(self.pool(self.relu(self.do2(self.conv2(x)))))
        # x = self.bn1(self.pool(self.relu(self.conv1(x))))
        # x = self.bn2(self.pool(self.relu(self.conv2(x))))
        x = self.flatten(self.pool(self.relu(self.conv3(x))))
        row = self.fc_row(x)
        col = self.fc_col(x)
        return row, col


train_set = DataLoader(Grid(X_train, y_train), batch_size=128, shuffle=True)
net = Net()
crit = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.01)
net.train()
n_epochs = 100
for epoch in range(1, n_epochs + 1):
    losses = []
    for x, y in train_set:
        opt.zero_grad()
        rows, cols = net(x)
        loss = crit(rows, y[:, 0])
        loss += crit(cols, y[:, 1])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("epoch", epoch, torch.tensor(losses).mean())


test_set = DataLoader(Grid(X_test, y_test), batch_size=len(X_test), shuffle=True)
net.eval()
# show_preds(net, test_set)
calc_exact(net, test_set)
calc_exact(net, train_set)


# TO DO - check eval with valid set
# min size of CNN
