import torch
import torch.nn as nn

from functions.data_manipulation import get_weights, get_gradient_weights

import time


class LSTMModel(nn.Module):
    # Inspiration: https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        dropout_prob,
        batch_size,
        device,
    ):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # TODO mean pooling layer? as in https://www.kaggle.com/vitaliykoren/lstm-with-two-dimensional-max-pooling-with-pytorch
        # self.mp = nn.AvgPool2d(4)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.set_device = device

        # Initializing hidden state for first input with zeros
        self.set_h0 = torch.zeros(
            layer_dim,
            batch_size,
            hidden_dim,
            requires_grad=True,
            device=device,
        )

        # Initializing cell state for first input with zeros
        self.set_c0 = torch.zeros(
            layer_dim,
            batch_size,
            hidden_dim,
            requires_grad=True,
            device=device,
        )

    def forward(
        self,
        x,
        jet_track_local,
        pooling="last",
        theta=None,
        theta_gradients=None,
        backpropagation_flag=True,
    ):
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        try:
            _, (hn, cn) = self.lstm(x, (self.set_h0.detach(), self.set_c0.detach()))
        except RuntimeError:
            # h0/c0 wrong dim -> torch doc: "Defaults to zeros if (h_0, c_0) is not provided."
            _, (hn, cn) = self.lstm(x)
        # TODO of mean pooling hier?
        # out = F.avg_pool2d(out)
        # self.mp(out)
        # self.mp(hn)


        # get mean/last pooled hidden states
        if pooling == "last":
            h_bar = hn[:, jet_track_local]
        elif pooling == "mean":
            h_bar = torch.zeros([hn.shape[0], len(jet_track_local), hn.shape[-1]]).to(
                self.set_device
            )
            jet_track_prev = 0
            jet_track_local = [x + 1 for x in jet_track_local]
            jet_track_local[-1] = None
            for j, jet_track in enumerate(jet_track_local):
                h_bar[:, j] = torch.mean(hn[:, jet_track_prev:jet_track], dim=1)
                jet_track_prev = jet_track

        # Check if backpropogation is required
        if backpropagation_flag:
            # Do backward to get gradients with respect to hn (to get first part of chain rule, only take derivative of kappa later for algorithm Tolga)
            hn.sum().backward()

            # Get parameters to update, save in dict for easy reference.
            if (
                not theta
            ):  # Use this condition to only get theta once, it doens't change.
                theta = get_weights(model=self.lstm, hidden_dim=hn.shape[2])

            theta_gradients = get_gradient_weights(
                model=self.lstm,
                hidden_dim=hn.shape[2],
                theta_gradients=theta_gradients,
            )

            return h_bar, theta, theta_gradients

        else:
            return h_bar
