import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.em = nn.Embedding(10, 1)
        self.rnn = nn.RNN(1, 10, nonlinearity="relu", batch_first=True)
        self.fc = nn.Linear(10, 10)

    def forward(self, x, h):
        x = self.em(x).squeeze(-1)
        x, h = self.rnn(x, h)
        x = self.fc(x)
        return x, h


def get_xy(size=100):
    digits = [i for i in range(10)]
    X = []
    y = []
    for i in range(size):
        x = random.sample(digits, 3)
        X.append(x)
        sorted_x = x.copy()
        sorted_x.sort()
        y.append(sorted_x)
    X = torch.tensor(X, dtype=torch.long).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


X_train, y_train = get_xy(10)
ds = TensorDataset(X_train, y_train)
dl = DataLoader(ds, batch_size=2)

net = Net()
opt = optim.Adam(net.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()
n_epochs = 1
for epoch in range(1, n_epochs + 1):
    losses = []
    for x, y in dl:
        print("###", x.shape)
        h = torch.randn(1, len(x), 10)
        opt.zero_grad()
        x, h = net(x, h)
        x = x.squeeze(0)
        print(x.shape)
        print(y.shape)
        loss = crit(x, y)
        # print(torch.argmax(x, dim=-1))
        # print(y)
        # print(loss)
        # loss = crit(rows, y[:, 0])
        # loss += crit(cols, y[:, 1])
        # loss.backward()
        # opt.step()
        # losses.append(loss.item())
    # print("epoch", epoch, torch.tensor(losses).mean())

