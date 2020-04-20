import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


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


def resave_for_resnet(data_path, batch_size=400):
    X = np.load(os.path.join(data_path, "X.npy"))
    X = X.reshape(-1, 3, 32, 32)
    for X, name in ((X[:50000], "X_train"), (X[50000:], "X_valid")):
        create_batches(X, batch_size, data_path, name)


def create_batches(X, batch_size, data_path, name):
    for i, idx in tqdm(enumerate(range(0, X.shape[0], batch_size))):
        new_batch = np.zeros((batch_size, 3, 224, 224), dtype="uint8")
        for j, sample in enumerate(X[idx : idx + batch_size]):
            new_batch[j] = resize_array(sample)
        b_name = name + "_" + str(i + 1) + ".npy"
        np.save(os.path.join(data_path, b_name), new_batch)


def resize_array(a):
    new_a = np.zeros((3, 256, 256), dtype="uint8")
    for i, comp in enumerate(a):
        new_a[i] = np.kron(comp, np.ones((8, 8), dtype="uint8"))
    return new_a[:, 16:240, 16:240]


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


class CIFARRES(CIFAR):
    def __init__(self, data_path, y, transform, valid=False):
        self.data_path = data_path
        self.valid = valid
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __getitem__(self, idx):
        new_idx = idx % 400
        if self.valid:
            i = int((idx / len(self)) * 25)
            name = f"X_valid_{i + 1}.npy"
        else:
            i = int((idx / len(self)) * 25)
            name = f"X_train_{i + 1}.npy"
        X = np.load(os.path.join(self.data_path, name)).astype("float32")
        x = torch.from_numpy(X[new_idx])
        return self.transform(x), self.y[idx]

    def __len__(self):
        if self.valid:
            return 10000
        else:
            return 10000


def get_datasets(data_path):
    X = np.load(os.path.join(data_path, "X.npy")).astype("float32")
    X = X.reshape(-1, 3, 32, 32)
    y = np.load(os.path.join(data_path, "y.npy")).astype("int64")
    xmean = np.mean(X, (0, 2, 3))
    xstd = np.std(X, (0, 2, 3))
    transform = transforms.Normalize(list(xmean), list(xstd))
    training_set = CIFAR(X[:-10000], y[:-10000], transform)
    validation_set = CIFAR(X[-10000:], y[-10000:], transform)
    mini_set = CIFAR(X[:1000], y[:1000], transform)
    return training_set, validation_set, mini_set


def get_datasets_res(data_path):
    X = np.load(os.path.join(data_path, "X.npy")).astype("float32")
    X = X.reshape(-1, 3, 32, 32)
    X = X / 255.0
    y = np.load(os.path.join(data_path, "y.npy")).astype("int64")
    xmean = (0.485, 0.456, 0.406)
    xstd = (0.229, 0.224, 0.225)
    transform = transforms.Normalize(xmean, xstd)
    training_set = CIFAR(X[:-10000], y[:-10000], transform)
    validation_set = CIFAR(X[-10000:], y[-10000:], transform)
    return training_set, validation_set


def plot(img, label=None):
    plt.imshow(np.transpose(img, axes=[1, 2, 0]))
    plt.axis("off")
    if label:
        plt.title(label)
    plt.plot()


def show_preds(net, dataset):
    with torch.no_grad():
        x_batch, y_batch = next(iter(dataset))
        p = net(x_batch)
        v, i = p.max(axis=1)
        for pred, true in zip(i, y_batch):
            print(f"pred {CIFAR.label_names[pred]}({CIFAR.label_names[true]})")
