import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

from functions.data_saver import save_results, save_loss_plots, DataTrackerTrials
from functions.data_selection import (
    train_validation_split,
    format_ak_to_list,
)
from functions.training import train_model_with_hyper_parameters, getBestModelfromTrials
from functions.data_loader import load_n_filter_data


from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials

import names as na
from ai.model import LSTM
from functions import data_loader
from functools import partial


# Variables
# File
file_name = "samples\JetToyHIResultSoftDropSkinny.root"

# Output dimension
output_dim = 3

# flags
flag_save_intermediate_results = False
flag_save_loss_plots = False
# hyper tuning space
max_evals = 1
space = hp.choice(
    "hyper_parameters",
    [
        {
            "num_batch": hp.choice("num_batch", [100, 1000]),
            "num_epochs": hp.choice("num_epochs", [1, 2]),
            "num_layers": hp.choice("num_layers", [1]),
            "hidden_size0": hp.choice("hidden_size0", [8]),
            "hidden_size1": hp.choice("hidden_size1", [4]),
            "learning_rate": hp.choice("learning_rate", [0.01]),
            "decay_factor": hp.choice("decay_factor", [0.9]),
            "loss_func": hp.choice("loss_func", ["mse"]),
        }
    ],
)

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)


# only use g_recur_jets
training_data, validation_data = train_validation_split(g_recur_jets, split=0.8)


# Setup datatracker for trials
data_tracker = DataTrackerTrials

# Find best model
trials = Trials()
best = fmin(
    partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
        train_model_with_hyper_parameters,
        training_data=training_data,
        validation_data=validation_data,
        data_tracker=data_tracker,
        output_dim=output_dim,
    ),
    space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)
print(space_eval(space, best))
