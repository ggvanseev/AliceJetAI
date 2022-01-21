import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from torch.nn.modules.loss import HingeEmbeddingLoss
from functions.data_manipulation import get_weights, get_gradient_weights
import numpy as np

from sklearn.svm import OneClassSVM


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
        # TODO mean pooling layer? as in https://www.kaggle.com/vitaliykoren/lstm-with-two-dimensional-max-pooling-with-pytorch
        self.mp = nn.AvgPool2d(4)

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
        # TODO size of out is now [210, 210]?

        # TODO of mean pooling hier?
        # out = F.avg_pool2d(out)
        # self.mp(out)
        # self.mp(hn)

        # Do backward to get gradients with respect to hn (to get first part of chain rule, only take derivative of kappa later for algorithm Tolga)
        hn.sum().backward()
        # [hn track_jet_select] # call backward ojn each jet output

        # Get parameters to update, save in dict for easy reference.
        theta = get_weights(model=self.lstm, hidden_dim=hn.shape[2])
        theta_gradients = get_gradient_weights(model=self.lstm, hidden_dim=hn.shape[2])

        # Convert the final state to our desired output shape (batch_size, output_dim) # retain_graph=True
        out = self.fc(out)

        return out, hn, theta, theta_gradients


def objective_function(alphas, h_bar):
    n_entries = len(alphas)
    matrix = 1


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

        # OneClass SVM
        self.ocsvm = OneClassSVM(kernel="linear")

    def train_step(self, x, y, jet_track):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat, hn, theta = self.model(x)

        # get mean pooled hidden states
        h_bar = hn[:, jet_track].cpu().detach().numpy()

        # a =[hn.T[x] for x in jet_track][0][i,:].cpu().detach().numpy() selects i-th "mean pooled output"
        # a.dot(a) = h.T * h = scalar

        # first need to run all samples to always have alphas match correctly. Thus fully run dataset through lstm before updating lstm parameters?

        # h_bar[0] @ h_bar[0].T instead of h_bar[0].T @ h_bar[0], to get matrix representation of h_bar_i.T*h_bar_j
        self.ocsvm.fit(h_bar[0])

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
            # track branch number for tracking what jet_track array to use
            i = 0
            for x_batch, y_batch in train_loader:
                jet_track_local = jet_track[i]  # Index where jets are
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
