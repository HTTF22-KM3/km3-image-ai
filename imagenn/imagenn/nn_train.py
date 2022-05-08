"""
This is the main script for the neural network that will later generate the extension.
"""

from __future__ import print_function

import os
import time

import numpy
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from datetime import datetime
from os import listdir


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


# Class for the NN; the magic happens here!
class NN (nn.Module):

    # Constructor
    def __init__(self):
        self.dtype = torch.float

        # Determines the device to compute on
        try:
            self.device = torch.device("cuda:0")
            print_debug_text("CUDA found: using GPU for computing")
        except RuntimeError:
            self.device = torch.device("cpu")
            print_debug_text("No CUDA found: using CPU for computing")

        # 3 Arrays:
        # self.amount_values stores the amount of indexes of the np arrays
        # self.train consists out of the (now shuffled) first 90% of temp
        # self.test consists out of the last 10% of temp
        path = "../data/"
        self.temp = []
        for file in os.listdir(path):
            self.temp.append(get_data(f"{path}/{file}"))

        np.random.shuffle(self.temp)

        self.train = self.temp[(len(self.temp)/9)*8:]
        self.test = self.temp[:len(self.temp)/9]

        del path
        del self.temp



