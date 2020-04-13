import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


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
    y = np.load(os.path.join(data_path, "y.npy"))
    xmean = np.mean(X, (0, 2, 3))
    xstd = np.std(X, (0, 2, 3))
    transform = transforms.Normalize(list(xmean), list(xstd))
    training_set = CIFAR(X[:-10000], y[:-10000], transform)
    validation_set = CIFAR(X[-10000:], y[-10000:], transform)
    return training_set, validation_set


def plot(img, label=None):
    plt.imshow(np.transpose(img, axes=[1, 2, 0]))
    plt.axis("off")
    if label:
        plt.title(label)
    plt.plot()
