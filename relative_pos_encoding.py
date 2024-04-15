import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from einops import rearrange
from einops.layers.torch import Rearrange


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=64, shuffle=False)

#train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
#test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

device = "mps" if torch.has_mps else "cpu"

def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training Loss of Absolute Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training Accuracy of Absolute Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()