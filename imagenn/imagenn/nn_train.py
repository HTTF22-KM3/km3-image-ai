"""
This is the main script for the neural network that will later generate the extension.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets

import pickle as pkl

from datetime import datetime


# Just a small function to make the code more readable. Prints out a text with the format
# Hour:Minute:Second.MillisecondMicrosecond | [file_name]: Text
def print_debug_text(text: str):
    print(f"{datetime.utcnow().hour}:{datetime.utcnow().minute}:{datetime.utcnow().second}."
          f"{datetime.utcnow().microsecond} | [nn_train.py]: {text}")


# Gets the data for the nn from a numpy array (Gets called in the NN-Class constructor)
def get_data(p: str):
    file = open(p, "r")
    array = pkl.load(file)
    file.close()
    return array


# Constructor
dtype = torch.float

# Determines the device to compute on
try:
    device = torch.device("cuda:0")
    print_debug_text("CUDA found: using GPU for computing")
except RuntimeError:
    device = torch.device("cpu")
    print_debug_text("No CUDA found: using CPU for computing")

path = "../data/"
batch_size = 64

transform = transforms.Compose([transforms.Resize(1000), transforms.ToTensor])

image_data = datasets.ImageFolder(path, transform=transform)

train_loader = torch.utils.data.DataLoader(image_data, batch_size, shuffle=True)


def scale(img, feature_range=(-1, 1)):
    minimum, maximum = feature_range
    img = img * (maximum-minimum) + minimum
    return img


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, stride, padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)