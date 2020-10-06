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
from prettytable import PrettyTable


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


def create_arrays(folder):
    """Convert pickle file to numpy arrays."""
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
    """CIFAR10 as torch dataset."""

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
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]

    def get_label(self, idx):
        label = self.y[idx]
        return self.label_names[int(label)]

    def get_image(self, idx):
        img = np.transpose(self.X[idx], axes=[0, 1, 2])
        return img.astype("int")


def get_datasets(data_path):
    X = np.load(os.path.join(data_path, "X.npy"))  # .astype("float32")
    X = X.reshape(-1, 3, 32, 32)
    X = np.transpose(X, axes=[0, 2, 3, 1])
    y = np.load(os.path.join(data_path, "y.npy")).astype("int64")
    xmean = np.mean(X, (0, 1, 2))
    xstd = np.std(X, (0, 1, 2))
    normalize = transforms.Normalize(list(xmean / 255), list(xstd / 255))
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(mode="RGB"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToPILImage(mode="RGB"), transforms.ToTensor(), normalize]
    )
    training_set = CIFAR(X[:-10000], y[:-10000], train_transform)
    validation_set = CIFAR(X[-10000:], y[-10000:], test_transform)
    mini_train_set = CIFAR(X[:1000], y[:1000], train_transform)
    mini_test_set = CIFAR(X[-1000:], y[-1000:], test_transform)
    return training_set, validation_set, mini_train_set, mini_test_set


def plot(img, label=None):
    plt.imshow(img)
    plt.axis("off")
    if label:
        plt.title(label)
    plt.plot()


def show_preds(net, dataset):
    table = PrettyTable(["Correct", "Prediction", "Correct Label"])
    with torch.no_grad():
        net.eval()
        x_batch, y_batch = next(iter(dataset))
        p = net(x_batch)
        v, i = p.max(axis=1)
        for pred, true in zip(i, y_batch):
            table.add_row(
                [
                    (pred == true).item(),
                    CIFAR.label_names[pred],
                    CIFAR.label_names[true],
                ]
            )
    print(table)


class ConvLayer(nn.Module):
    """2dConv Layer with Batch normalization."""

    kernel_size = (3, 3)
    padding = 1
    bias = False
    stride = 1

    def __init__(self, input_size, output_size, stride=None):
        super().__init__()
        stride = stride if stride else self.stride
        self.conv = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, x):
        return self.bn(self.conv(x))


class ShortcutConv(ConvLayer):
    """Shortcut with 1x1 Conv layer instead of padding with 0s."""

    kernel_size = (1, 1)
    padding = 0
    bias = False
    stride = 2


class RESNET(nn.Module):
    """Resnet for CIFAR10 implemented the same as in original paper except for shortcut resize."""

    filter_groups = [16, 32, 64]

    def __init__(self, n_blocks):
        super().__init__()
        current_filters = self.filter_groups[0]
        conv = ConvLayer(3, current_filters, stride=1)
        relu = nn.ReLU(inplace=True)
        layers = [conv, relu]
        for n_filters in self.filter_groups:
            for block in range(n_blocks):
                layers.append(RESBLOCK(current_filters, n_filters))
                current_filters = n_filters
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RESBLOCK(nn.Module):
    """Standard Resblock. """

    def __init__(self, n_in, n_out):
        super().__init__()
        self.subsample = True if n_in != n_out else False
        initial_stride = 2 if self.subsample else 1
        self.shortcut_conv = ShortcutConv(n_in, n_out) if self.subsample else None
        self.conv1 = ConvLayer(n_in, n_out, stride=initial_stride)
        self.conv2 = ConvLayer(n_out, n_out)

    def forward(self, x):
        shortcut = self.shortcut_conv(x) if self.subsample else x
        x = F.relu(self.conv1(x))
        x = F.relu(shortcut + self.conv2(x))
        return x
