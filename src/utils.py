import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

def get_transforms():

        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels =3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

def calculate_accuracy(y_pred, y_true):
    _, preds = torch.max(y_pred, 1)
    return torch.sum(preds == y_true).item()


def plot_training(train_losses, val_losses, val_acc):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val loss")
    plt.legend()
    plt.title("Loss")


    plt.subplot(1,2,2)
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.show()