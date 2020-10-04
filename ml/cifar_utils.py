import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from utils import get_output_shape


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


def create_arrays(folder):
    PTN = "_batch"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if PTN in f]
    data = []
    labels = []
    for f in files:
        d = unpickle(f)
        data.append(d[b"data"])
        labels.append(d[b"labels"])
    for filename, d in zip(["X.npy", "y.npy"], [data, labels]):
        array = np.concatenate(d)
        path = os.path.join(folder, filename)
        np.save(path, array)


class CIFAR(Dataset):

    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, X, y, transform):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]

    def get_label(self, idx):
        label = self.y[idx]
        return self.label_names[int(label)]

    def get_image(self, idx):
        img = self.X[idx].numpy()
        return img.astype("int")


def get_datasets(data_path):
    X = np.load(os.path.join(data_path, "X.npy")).astype("float32")
    X = X.reshape(-1, 3, 32, 32)
    y = np.load(os.path.join(data_path, "y.npy")).astype("int64")
    xmean = np.mean(X, (0, 2, 3))
    xstd = np.std(X, (0, 2, 3))
    transform = transforms.Normalize(list(xmean), list(xstd))
    training_set = CIFAR(X[:-10000], y[:-10000], transform)
    validation_set = CIFAR(X[-10000:], y[-10000:], transform)
    mini_train_set = CIFAR(X[:1000], y[:1000], transform)
    mini_test_set = CIFAR(X[-1000:], y[-1000:], transform)
    return training_set, validation_set, mini_train_set, mini_test_set


def plot(img, label=None):
    plt.imshow(np.transpose(img, axes=[1, 2, 0]))
    plt.axis("off")
    if label:
        plt.title(label)
    plt.plot()


def show_preds(net, dataset):
    with torch.no_grad():
        net.eval()
        x_batch, y_batch = next(iter(dataset))
        p = net(x_batch)
        v, i = p.max(axis=1)
        for pred, true in zip(i, y_batch):
            print(f"pred {CIFAR.label_names[pred]}({CIFAR.label_names[true]})")


class RESNET(nn.Module):
    """Resnet for CIFAR10 implemented the same as in original paper except for shortcut connections."""

    filter_groups = [16, 32, 64]

    def __init__(self, n_blocks):
        super().__init__()
        current_filters = self.filter_groups[0]
        conv1 = nn.Conv2d(
            current_filters,
            current_filters,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        bn1 = nn.BatchNorm2d(current_filters)
        relu = nn.ReLU(inplace=True)
        layers = [conv1, bn1, relu]
        for n_filters in self.filter_groups:
            for block in range(n_blocks):
                layers.append(RESBLOCK(current_filters, n_filters))
                current_filters = n_filters
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten)
        layers.append(nn.Linear(64, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RESBLOCK(nn.Module):
    """Standard Resblock. Shortcut done with 1x1 Conv layer instead of padding with 0s."""

    def __init__(self, n_filters_in, n_filters_out):
        super().__init__()
        self.subsample = True if n_filters_in != n_filters_out else False
        initial_stride = 2 if self.subsample else 1
        self.shortcut = (
            nn.Conv2d(
                n_filters_in,
                n_filters_out,
                kernel_size=(1, 1),
                stride=2,
                padding=0,
                bias=False,
            )
            if self.subsample
            else None
        )
        self.conv1 = nn.Conv2d(
            n_filters_in,
            n_filters_out,
            kernel_size=(3, 3),
            stride=initial_stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(n_filters_out)
        self.conv2 = nn.Conv2d(
            n_filters_out,
            n_filters_out,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(n_filters_out)

    def forward(self, x):
        if self.subsample:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(shortcut + (self.bn2(self.conv2(x))))
        return x
