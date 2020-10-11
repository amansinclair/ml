import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score as acc
from prettytable import PrettyTable


def count_parameters(model, show_table=False):
    """Taken from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if show_table:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def accuracy(labels, probs):
    v, i = probs.max(axis=1)
    size = len(labels)
    n_true = (i == labels).sum().item()
    return n_true / size


def top_n(probs, n=3):
    s, idxes = torch.sort(probs, descending=True)
    return idxes[:, :n]


def top_n_accuracy(labels, probs, n=3):
    size = len(labels)
    total = 0
    top_probs = top_n(probs, n)
    for col in range(n):
        total += (top_probs[:, col] == labels).sum().item()
    return total / size


def train(net, crit, opt, train, val=None, metric=None, n_epochs=1):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        pbar = tqdm(train)
        for X, y in pbar:
            opt.zero_grad()
            net.train()
            p = net(X)
            if metric:
                epoch_acc.append(metric(y, p))
            loss = crit(p, y)
            epoch_loss.append(loss.item())
            loss.backward()
            opt.step()
        if val:
            with torch.no_grad():
                net.eval()
                val_loss = []
                val_acc = []
                for X, y in val:
                    p = net(X)
                    val_loss.append(crit(p, y).item())
                    if metric:
                        val_acc.append(metric(y, p))
            test_loss.append(np.mean(val_loss))
            test_acc.append(np.mean(val_acc))
        train_loss.append(np.mean(epoch_loss))
        train_acc.append(np.mean(epoch_acc))
        msg = f"Epoch:{epoch + 1}, T Loss:{np.mean(epoch_loss):.3f}"
        if metric:
            msg += f", T Met:{np.mean(epoch_acc):.3f}"
        if val:
            msg += f", V Loss:{np.mean(val_loss):.3f}"
            if metric:
                msg += f", V Met:{np.mean(val_acc):.3f}"
        print(msg)
    return train_loss, train_acc, test_loss, test_acc


def plot_loss_acc(train_loss, train_acc, test_loss, test_acc):
    epochs = np.arange(len(train_loss))
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, test_loss, label="test")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, train_acc, label="train")
    ax[1].plot(epochs, test_acc, label="test")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    plt.show()


def plot_loss(train_loss, test_loss):
    epochs = np.arange(len(train_loss))
    plt.figure(figsize=(15, 8))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, test_loss, label="test")
    plt.legend()
    plt.show()


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

