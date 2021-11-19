"""
Training the LSTM.
"""
from typing import no_type_check

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import uproot
import awkward as ak
import pandas as pd

import names as na
from ai.model import *
from ai.dataset import *
from functions import data_loader

# load root dataset into a pandas dataframe
fileName = "./samples/JetToyHIResultSoftDropSkinny.root"
g_jets, q_jets, g_recur_jets, q_recur_jets = data_loader.load_n_filter_data(fileName)

print(ak.to_pandas(g_jets).head(), len(ak.to_pandas(g_jets)), sep="\n")
print(ak.to_pandas(g_recur_jets).head(), len(ak.to_pandas(g_recur_jets)), sep="\n")

# Recursive gluon jet data in list form
g_list = data_loader.format_ak_to_list(g_recur_jets)

# check if gpu is available, otherwise use cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

batch_size = 64
train_data = JetDataset(g_list[:700])
test_data = JetDataset(g_list[700:1000])
train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)
print("hi")


def train(n_epochs: int, lr: float = 0.04):

    # build a model
    model = LSTM(3, 4, 4, 3, device)

    # optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(n_epochs):
        print("yea")
    return


train(3)
