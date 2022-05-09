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


# TODO: Get behind the math of this algorithm
# TODO: Read paper for Discriminative AI and Generative AI
class Discriminator():
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim

        self.cv1 = conv(3, self.conv_dim, 4, batch_norm=False)
        self.cv2 = conv(self.conv_dim, self.conv_dim*2, 4, batch_norm=True)
        self.cv3 = conv(self.conv_dim*2, self.conv_dim*4, batch_norm=True)
        self.cv4 = conv(self.conv_dim * 4, self.conv_dim * 8, batch_norm=True)

        self.fc1 = nn.Linear(self.conv_dim*8*2*2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x), 0.2)
        x = F.leaky_relu(self.cv2(x), 0.2)
        x = F.leaky_relu(self.cv3(x), 0.2)
        x = F.leaky_relu(self.cv4(x), 0.2)

        x = x.view(-1, self.conv_dim*8*2*2)
        x = self.fc1(x)
        return x


# Helper function for generative AI
def deconv(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    layers = []
    convt_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    layers.append(convt_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.z_size = z_size
        self.conv_dim = conv_dim

        # Declaring the fully connected layer
        self.fc = nn.Linear(z_size, self.conv_dim*8*2*2)
        self.dcv1 = deconv(self.conv_dim*8, self.conv_dim*4, 4, batch_norm=True)
        self.dcv2 = deconv(self.conv_dim*4, self.conv_dim*2, 4, batch_norm=True)
        self.dcv3 = deconv(self.conv_dim*2, self.conv_dim, 4, batch_norm=True)
        self.dcv4 = deconv(self.conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2)

        x = F.relu(self.dcv1(x))
        x = F.relu(self.dcv2(x))
        x = F.relu(self.dcv3(x))
        x = F.relu(self.dcv4(x))

        return x
