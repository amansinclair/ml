import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as bar
from sklearn.metrics import accuracy_score as acc
from prettytable import PrettyTable


def count_parameters(model):
    """Taken from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(net, crit, acc, opt, train, val, n_epochs=1):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        for X, y in bar(train):
            opt.zero_grad()
            net.train()
            p = net(X)
            v, i = p.max(axis=1)
            epoch_acc.append(acc(y, i))
            loss = crit(p, y)
            epoch_loss.append(loss.item())
            loss.backward()
            opt.step()
        with torch.no_grad():
            net.eval()
            val_loss = []
            val_acc = []
            for X, y in val:
                p = net(X)
                v, i = p.max(axis=1)
                val_loss.append(crit(p, y).item())
                val_acc.append(acc(y, i))
        test_loss.append(np.mean(val_loss))
        test_acc.append(np.mean(val_acc))
        train_loss.append(np.mean(epoch_loss))
        train_acc.append(np.mean(epoch_acc))
        print(
            f"Epoch:{epoch + 1}, T Loss:{np.mean(epoch_loss):.3f}, T acc:{np.mean(epoch_acc):.3f} V Loss:{np.mean(val_loss):.3f}, V acc:{np.mean(val_acc):.3f}"
        )
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


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape
