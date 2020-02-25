import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 6, 3)

    def forward(self, x):
        x = self.conv1(x)


net = Net()

for param in net.parameters():
    print(param.data.shape)
