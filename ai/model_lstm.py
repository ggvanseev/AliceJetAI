import torch
import torch.nn as nn

from functions.data_manipulation import get_weights, get_gradient_weights


class LSTMModel(nn.Module):
    # Inspiration: https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # TODO mean pooling layer? as in https://www.kaggle.com/vitaliykoren/lstm-with-two-dimensional-max-pooling-with-pytorch
        self.mp = nn.AvgPool2d(4)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        backpropagation_flag=True,
    ):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(
            self.layer_dim,
            x.size(0),
            self.hidden_dim,
            requires_grad=True,
            device=device,
        )

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(
            self.layer_dim,
            x.size(0),
            self.hidden_dim,
            requires_grad=True,
            device=device,
        )

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # TODO of mean pooling hier?
        # out = F.avg_pool2d(out)
        # self.mp(out)
        # self.mp(hn)

        # Check if backpropogation is required
        if backpropagation_flag:
            # Do backward to get gradients with respect to hn (to get first part of chain rule, only take derivative of kappa later for algorithm Tolga)
            hn.sum().backward()
            # [hn track_jet_select] # call backward ojn each jet output

            # Get parameters to update, save in dict for easy reference.
            theta = get_weights(model=self.lstm, hidden_dim=hn.shape[2])
            theta_gradients = get_gradient_weights(
                model=self.lstm, hidden_dim=hn.shape[2]
            )

            return hn, theta, theta_gradients

        else:
            return hn
