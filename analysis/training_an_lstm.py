import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

from functions.data_saver import save_results, save_loss_plots, DataTrackerTrials
from functions.training import train_model_with_hyper_parameters, getBestModelfromTrials

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials

import names as na
from ai.model import LSTM
from functions import data_loader
from functools import partial


# Variables
# flags
flag_save_intermediate_results = False
flag_save_loss_plots = False
# hyper tuning space
max_evals = 1
space = hp.choice(
    "hyper_parameters",
    [
        {
            "num_batch": hp.quniform("num_batch", 10000, 20000, 2000),
            "num_epochs": hp.quniform("num_epochs", 30, 50, 5),
            "num_layers": hp.quniform("num_layers", 2, 4, 1),
            "hidden_size0": hp.quniform("hidden_size0", 8, 20, 2),
            "hidden_size1": hp.quniform("hidden_size1", 4, 8, 2),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
            "decay_factor": hp.uniform("decay_factor", 0.9, 0.99),
            "loss_func": hp.choice("loss_func", ["mse"]),
        }
    ],
)

# samples

training_data = Samples(
    "./data/Training/jewel_R_pt120_zcut0p1_beta0_mult7000.root",
    "csejet",
    [1.0, 0.0],
    [0, 300000],
)

validation_data = Samples(
    "./data/Validation/jewel_R_pt120_zcut0p1_beta0_mult7000.root",
    "csejet",
    [1.0, 0.0],
    [0, 150000],
)


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
    ),
    space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)
print(space_eval(space, best))
