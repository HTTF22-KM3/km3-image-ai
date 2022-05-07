"""
This is the main script for the neural network that will later generate the extension.
"""

import torch
import numpy as np
import pickle as pkl

data_path = ""


# Read the data from the numpy array (saved as pickle).
# Returns the Tensors train and test for later use
def grab_data(f: str):
    file = open(f, "r")
    temp = pkl.load(file)
    file.close()
    np.random.shuffle(temp)
    test_tensor: torch.Tensor = torch.Tensor(temp[:(len(temp)/9)])
    train_tensor: torch.Tensor = torch.Tensor(temp[temp-len(test):])

    return train_tensor, test_tensor


# Main train-function. Used for training the model (duh!)
def train():
    pass
