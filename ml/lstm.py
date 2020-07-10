import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import random


def get_curve(size=1000):
    e = np.random.randn(size) * 0.01
    lin = np.arange(size) * 0.03
    sin = np.array([math.sin(math.radians(x * 30)) for x in range(size)])
    return (e + lin + sin).astype("float32")


def get_xy(curve, seq_len):
    start = np.random.randint(1, seq_len - 1)
    x_segs = []
    y_segs = []
    for i in range((len(curve) - start) // seq_len):
        x_segs.append(torch.from_numpy(curve[start : start + seq_len]))
        y_segs.append(torch.from_numpy(curve[start + 1 : start + seq_len + 1]))
        start += seq_len
    X = torch.stack(x_segs)
    y = torch.stack(y_segs)
    return X.view(-1, seq_len, 1), y.view(-1, seq_len, 1)


class Net(nn.Module):
    def __init__(self, size=16):
        super().__init__()
        self.lstm = nn.LSTM(1, size, batch_first=True)
        self.fc = nn.Linear(size, 1)

    def forward(self, x, cell_stuff):
        # print(x.shape, cell_stuff[0].shape, cell_stuff[1].shape)
        x, cell_stuff = self.lstm(x, cell_stuff)
        return self.fc(x), cell_stuff


seq_len = 40
curve = get_curve(2000)
m = curve.mean()
std = curve.std()
curve = (curve - m) / std

size = 8
net = Net(size)
crit = nn.MSELoss()
optim = optim.Adam(net.parameters(), lr=0.05)
n_epochs = 200
batch_size = 5
X, y = get_xy(curve, seq_len)

for epoch in range(n_epochs):
    X, y = get_xy(curve, seq_len)
    n_segments = X.shape[0]
    idx = 0
    for i in range(n_segments // batch_size):
        xb = X[idx : idx + batch_size]
        yb = y[idx : idx + batch_size]
        idx += batch_size
        losses = []
        hn = torch.randn(1, batch_size, size)
        cn = torch.randn(1, batch_size, size)
        optim.zero_grad()
        output, (hn, cn) = net(xb, (hn, cn))
        loss = crit(output, yb)
        loss.backward()
        optim.step()
        losses.append(loss)
    print(f"epoch {epoch + 1}: {torch.stack(losses).mean()}")
with torch.no_grad():
    hn = torch.randn(1, X.shape[0], size)
    cn = torch.randn(1, X.shape[0], size)
    yt, crap = net(X, (hn, cn))
    plt.plot(yt.view(-1)[750:1250], "x")
    plt.plot(y.view(-1)[750:1250], "o")
    plt.show()
    # print("LOSS", crit(yt, yb))
    # print("REF LOSS", crit(xb, yb))
    # yt = yt.view(2, seq_len)
    # print(yt.shape)
    # plt.plot(yt[0], "x")
    # yb = yb.view(2, seq_len)
    # plt.plot(yb[0], "o")
    # plt.plot(xb[0], "+")
    # plt.show()

