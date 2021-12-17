import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib
# import joblib

import time


import torch.optim as optim
import torch.nn as nn

from functions.data_saver import save_results, DataTrackerTrials
from functions.data_manipulation import (
    train_dev_test_split,
    format_ak_to_list,
    branch_filler,
    lstm_data_prep,
)
from functions.data_loader import load_n_filter_data

from ai.model_lstm import LSTMModel, Optimization

# Variables:
batch_size = 210

output_dim = 1
layer_dim = 1
dropout = 0.2
n_epochs = 4
learning_rate = 1e-3
weight_decay = 1e-6

# File
# file_name = "samples/JetToyHIResultSoftDropSkinny.root" # Windows?
file_name = "samples/JetToyHIResultSoftDropSkinny.root" # Unix


# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)


# only use g_recur_jets
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])

train_data, track_jets_train_data = branch_filler(train_data, batch_size=batch_size)
dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)

# Only use train and dev data for now
scaler = (
    MinMaxScaler()
)  # Note this has to be saved with the model, to ensure data has the same form.
train_loader = lstm_data_prep(
    data=train_data, scaler=scaler, batch_size=batch_size, fit_flag=True
)
val_loader = lstm_data_prep(data=dev_data, scaler=scaler, batch_size=batch_size)

input_dim = len(train_data[0])
hidden_dim = batch_size

model_params = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "output_dim": output_dim,
    "dropout_prob": dropout,
}

model = LSTMModel(**model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(
    train_loader,
    val_loader,
    jet_track=track_jets_train_data,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_features=input_dim,
)

print(opt.model)

"""
Save scalar using:
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 

# And now to load...

scaler = joblib.load(scaler_filename) 
"""
