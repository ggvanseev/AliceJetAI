from typing import no_type_check

import torch

# import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

import names as na
from ai.model import LSTM
from functions import data_loader


# load root dataset into a pandas dataframe
fileName = "./samples/JetToyHIResultSoftDropSkinny.root"
g_jets, q_jets, g_recur_jets, q_recur_jets = data_loader.load_n_filter_data(
    fileName, na.tree
)


# check if gpu is available, otherwise use cpu
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_model(training_data, model, num_epochs, learning_rate, decay_factor):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate decay exponentially
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, decay_factor, last_epoch=-1
    )

    # training
    step_training = []
    loss_training = []
