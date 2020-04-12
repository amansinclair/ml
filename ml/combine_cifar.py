import os
import numpy as np
import pickle


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


PATH = ".//ml//data//cifar-10-batches-py"
PTN = "_batch"

files = [os.path.join(PATH, f) for f in os.listdir(PATH) if PTN in f]
data = []
labels = []

for f in files:
    d = unpickle(f)
    data.append(d[b"data"])
    labels.append(d[b"labels"])

X = np.concatenate(data)
y = np.concatenate(labels)
X_path = os.path.join(".", "ml", "data", "cifar", "X.npy")
y_path = os.path.join(".", "ml", "data", "cifar", "y.npy")
np.save(X_path, X)
np.save(y_path, y)
