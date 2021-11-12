import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from datetime import datetime

import torch.optim as optim

from functions.data_saver import save_results, DataTrackerTrials
from functions.data_manipulation import (
    train_dev_test_split,
    format_ak_to_list,
    branch_filler,
)

from functions.data_loader import load_n_filter_data

# Variables:
batch_size = 210

# File
file_name = "samples\JetToyHIResultSoftDropSkinny.root"


# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)


# only use g_recur_jets
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])

train_data, track_jets_train_data = branch_filler(train_data, batch_size=batch_size)
dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)

# Only use train and dev data for now
scaler = MinMaxScaler()
train_arr = scaler.fit_transform(train_data)
dev_arr = scaler.transform(dev_data)
# test_arr = scaler.transform(test_data)


train_features = torch.Tensor(train_data)
train_targets = torch.Tensor(train_data)
val_features = torch.Tensor(dev_arr)
val_targets = torch.Tensor(dev_arr)
# test_features = torch.Tensor(X_test_arr)
# test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
# test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out, hn


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y, jet_track):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat, hn = self.model(x)

        # a =[hn.T[x] for x in jet_track][0][i,:].cpu().detach().numpy() selects i-th "mean pooled output"
        # a.dot(a) = h.T * h = scalar

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(
        self,
        train_loader,
        val_loader,
        jet_track,
        batch_size=64,
        n_epochs=50,
        n_features=1,
    ):
        model_path = (
            f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            # track branch number
            i = 0
            for x_batch, y_batch in train_loader:
                jet_track_local = jet_track[i]
                i += 1
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch, jet_track_local)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values


input_dim = len(train_data[0])
output_dim = 3
hidden_dim = batch_size
layer_dim = 3
dropout = 0.2
n_epochs = 4
learning_rate = 1e-3
weight_decay = 1e-6

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
