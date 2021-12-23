"""
Using algorithm 1 of Unsupervised Anomaly Detection With LSTM Neural Networks
Sauce: Tolga Ergen and Suleyman Serdar Kozat, Senior Member, IEEE

-----------------------------------------------------------------------------------------
Algorithm 1: Quadratic Programming-Based Training for the Anomaly Detection Algorithm
             Based on OC-SVM
-----------------------------------------------------------------------------------------
1. Initialize the LSTM parameters as θ_0 and the dual OC-SVM parameters as α_0
2. Determine a threshold ϵ as convergence criterion
3. k = −1
4. do
5.    k = k+1
6.    Using θ_k, obtain {h}^n_{i=1} according to Fig. 2
7.    Find optimal α_{k+1} for {h}^n_{i=1} using (20) and (21)
8.    Based on α_{k+1}, obtain θ_{k+1} using (24) and Remark 3
8. while (κ(θ_{k+1}, α{k+1})− κ(θ_k, α))^2 > ϵ
9. Detect anomalies using (19) evaluated at θ_k and α_k
-----------------------------------------------------------------------------------------

(20): α_1 = 1 − S − α_2, where S= sum^n_{i=3} α_i.
(21): α_{k+1,2} = ((α_{k,1} + α_{k,2})(K_{11} − K_{12})  + M_1 − M_2) / (K_{11} + K_{22}
                                                                               − 2K_{12})
      K_{ij} =def= h ^T_iT h _j, Mi =def= sum^n_{j=3} α_{k,j}K_{ij}
(24): W^(·)_{k+1} = (I + (mu/2)A_k)^-1 (I− (mu/2)A_k) W^(·)_k
      Ak = Gk(W(·))T −W(·)GT

Dual problem of the OC-SVM:
(22): min_{theta}  κ(θ, α_{k+1}) = 1/2 sum^n_{i=1} sum^n_{j=1} α_{k+1,i} α_{k+1,j} h^T_i h_j
(23): s.t.: W(·)^T W(·) = I, R(·)^T R(·) = I and b(·)^T b(·) = 1

Remark 3: For R(·) and b(·), we first compute the gradient of the objective function with
respect to the chosen parameter as in (25). We then obtain Ak according to the chosen
parameter. Using Ak, we update the chosen parameter as in (24).

         ----------------------------------------------------------------------


Using algorithm 2 of Unsupervised Anomaly Detection With LSTM Neural Networks
Sauce: Tolga Ergen and Suleyman Serdar Kozat, Senior Member, IEEE]
"""

import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib
# import joblib

import time
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn

from functions.data_saver import save_results, DataTrackerTrials
from functions.data_manipulation import (
    train_dev_test_split,
    format_ak_to_list,
    branch_filler,
    lstm_data_prep,
    get_full_pytorch_weight,
    put_weight_in_pytorch_matrix,
)
from functions.data_loader import load_n_filter_data
from functions.optimization_orthogonality_constraints import optimization

from ai.model_lstm import LSTMModel

# from autograd import elementwise_grad as egrad

from copy import copy


file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# Variables:
batch_size = 100

output_dim = 1
layer_dim = 1
dropout = 0.2
n_epochs = 4
learning_rate = 1e-3
weight_decay = 1e-6

eps = 1e-2  # test value for convergence

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

# lstm model
lstm_model = LSTMModel(**model_params)

# svm model
svm_model = OneClassSVM(nu=0.5, gamma=0.35, kernel="rbf")

# path for model - only used for saving
# model_path = f'models/{lstm_model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def lstm_results(lstm_model, train_loader):
    """Obtain h_bar states from the lstm with the data

    Args:
        lstm_model (LSTMModel): the LSTM model
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch, stores
                the data

    Returns:
        torch.Tensor: contains the h_bar results from the LSTM
        dict: new theta containing weights and biases
    """
    h_bar_list = []

    i = 0
    for x_batch, y_batch in train_loader:
        jet_track_local = track_jets_train_data[i]
        i += 1

        x_batch = x_batch.view([batch_size, -1, model_params["input_dim"]]).to(device)
        y_batch = y_batch.to(device)

        ### Train step
        # set model to train
        lstm_model.train()

        # Makes predictions
        _, hn, theta, theta_gradients_temp = lstm_model(x_batch)

        if "theta_gradients" not in locals():
            theta_gradients = theta_gradients_temp
        else:
            for key1, value in theta_gradients_temp.items():
                for key2, value2 in value.items():
                    theta_gradients[key1][key2] = theta_gradients[key1][key2] + value2

        # get mean pooled hidden states
        h_bar = hn[:, jet_track_local]

        # h_bar_list.append(h_bar) # TODO, h_bar is not of fixed length! solution now: append all to list, then vstack the list to get 2 axis structure
        h_bar_list.append(h_bar)

    return torch.vstack([h_bar[0] for h_bar in h_bar_list]), theta


def kappa(alphas, a_idx, h_list):
    """Cost function to be minimized. Follows from the definitions of the OC-SVM.

    Args:
        alphas (numpy.ndarray): contains non-zero alpha values obtained from the SVM
                                                              with the SMO algorithm
        a_idx (numpy.ndarray): contains the indices of datapoints corresponding to the
                                                                 non-zero alpha values
        h_list (iterable): contains the h_bar results from the LSTM

    Returns:
        (torch.Tensor): kappa value resulting from equation (22) in Tolga's paper
    """
    out = 0
    for idx1, i in enumerate(a_idx):
        for idx2, j in enumerate(a_idx):
            out += 0.5 * alphas[0, idx1] * alphas[0, idx2] * (h_list[i].T @ h_list[j])
    return out


def delta_func(
    lstm_model,
    train_loader,
    h_list,
    weight,
    weight_name: str,
    mu,
    alphas,
    a_idx,
    pytorch_weights,
):
    """

    Args:
        lstm_model (LSTMModel): the LSTM model
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch, stores
                the data 
        h_list (torch.Tensor): contains the h_bar results from the LSTM
        weight (torch.Tensor): contains weights/biases of the LSTM
        weight_name (str): description of which weight/bias is currently used
        mu (float): learning rate
        alphas (numpy.ndarray): contains non-zero alpha values obtained from the SVM
                                                              with the SMO algorithm
        a_idx (numpy.ndarray): contains the indices of datapoints corresponding to the
                                                                 non-zero alpha values
        pytorch_weights (torch.Tensor): tensor 

    Returns:
        (torch.Tensor): derivative of the cost function to the weight/bias 
    """

    delta = mu * weight
    new_weight = weight - delta

    # Use torch.no_grad to not record changes in this section
    with torch.no_grad():

        lstm_model_new = copy(
            lstm_model  # Needs a copy, to avoid unexpected changes in the original model
        )

        # only updated desired weight element
        pytorch_weights = put_weight_in_pytorch_matrix(
            new_weight, weight_name, pytorch_weights
        )

        getattr(lstm_model_new.lstm, weight_name[5:]).copy_(pytorch_weights)

    h_list_new, _ = lstm_results(lstm_model_new, train_loader)
    return (
        kappa(alphas, a_idx, h_list) - kappa(alphas, a_idx, h_list_new)
    ) / delta  # Alphas, en a_idx worden niet correct geimporteerd in de functie


def updating_theta(lstm_model, train_loader, h_list, theta: dict, mu, alphas, a_idx):
    """Updates the weights of the LSTM contained in theta according to the optimization 
    algorithm with orthogonality constraints

    Args:
        lstm_model (LSTMModel): the LSTM model
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch, stores
                                                                                the data 
        h_list (torch.Tensor): contains the h_bar results from the LSTM
        theta (dict): contains the weights and bias values of the LSTM
        mu (float): learning rate
        alphas (numpy.ndarray): contains non-zero alpha values obtained from the SVM
                                                              with the SMO algorithm
        a_idx (numpy.ndarray): contains the indices of datapoints corresponding to the
                                                                 non-zero alpha values

    Returns:
        (dict): contains the updated weights and bias values of the LSTM
    """
    updated_theta = dict()

    # Loop over all weight types (w,r,bi,bh)
    for weight_type, weights in theta.items():

        track_weights = dict()

        # get full original weights, called pytorch_weights (following lstm structure)
        pytorch_weights = get_full_pytorch_weight(weights)

        for weight_name, weight in weights.items():
            # follow steps from eq. 24 in paper Tolga

            pytorch_weights_layer = pytorch_weights[weight_name[-1]]

            # derivative of function e.g. F = (25) from Tolga
            g = delta_func(
                lstm_model,
                train_loader,
                h_list,
                weight,
                weight_name,
                mu,
                alphas,
                a_idx,
                pytorch_weights_layer,
            )

            a = g @ weight.T - weight @ g.T
            i = torch.eye(weight.shape[0])

            # next point from Crank-Nicolson-like scheme
            track_weights[weight_name] = (
                torch.inverse(i + mu / 2 * a) @ (i - mu / 2 * a) @ weight
            )

        # store in theta
        updated_theta[weight_type] = track_weights

    return updated_theta


def update_lstm(lstm, theta):
    """Function to update the weights and bias values of the LSTM

    Args:
        lstm (LSTMModel): the LSTM model
        theta (dict): contains the weights and bias values of the LSTM

    Returns:
        lstm (LSTMModel): the updated LSTM model
    """
    for weight_type, weights in theta.items():
        # get full original weights, called pytorch_weights (following lstm structure)
        pytorch_weights = get_full_pytorch_weight(weights)

        weight_name = list(weights.keys())[0][5:-1]

        for i in range(len(weights) // 4):
            with torch.no_grad():
                getattr(lstm.lstm, weight_name + str(i)).copy_(pytorch_weights[str(i)])

    return lstm


def optimization(lstm, train_loader, alphas, a_idx, mu, h_list, theta):
    """Optimization algorithm with orthogonality constraints

    Args:
        lstm (LSTMModel): the LSTM model used
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch,
                                                                  stores the data 
        alphas (numpy.ndarray): contains non-zero alpha values obtained from the SVM
                                                              with the SMO algorithm
        a_idx (numpy.ndarray): contains the indices of datapoints corresponding to the
                                                                 non-zero alpha values
        mu (float): learning rate
        h_list (torch.Tensor): contains the h_bar results from the LSTM
        theta ([dict): contains the weights and bias values of the LSTM

    Returns:
        lstm (LSTMModel): the updated LSTM model
    """
    # update theta
    theta = updating_theta(lstm, train_loader, h_list, theta, mu, alphas, a_idx)

    # update lstm
    lstm = update_lstm(lstm, theta)

    return lstm, theta


### ALGORITHM START ###

# calculate cost function kappa with alpha_0 and theta_0

# obtain h_bar from the lstm with theta_0, given the data
h_bar_list, theta = lstm_results(lstm_model, train_loader)
h_bar_list_np = np.array([h_bar.detach().numpy() for h_bar in h_bar_list])

"""
alphas = np.abs(svm_model.dual_coef_)
a_idx = svm_model.support_"""
# TODO set alphas to 1 or 0 so cost_next - cost will be large

cost = 1  # kappa(alphas, a_idx, h_bar_list)


k = -1
while k < 5:  # TODO, (kappa(theta_next, alpha_next) - kappa(theta, alpha) < eps)

    # track branch number for tracking what jet_track array to use
    k += 1

    # keep previous cost result stored
    cost_prev = copy(cost)

    # obtain alpha_k+1 from the h_bars with SMO through the OC-SVMs .fit()
    svm_model.fit(h_bar_list_np)
    alphas = np.abs(svm_model.dual_coef_)
    a_idx = svm_model.support_

    # obtain theta_k+1 using the optimization algorithm
    lstm_model, theta_next = optimization(
        lstm_model,
        train_loader,
        alphas,
        a_idx,
        learning_rate,
        h_list=h_bar_list,
        theta=theta,
    )

    # obtain h_bar from the lstm with theta_k+1, given the data
    h_bar_list, theta = lstm_results(lstm_model, train_loader)
    h_bar_list_np = np.array([h_bar.detach().numpy() for h_bar in h_bar_list])

    # obtain the new cost given theta_k+1 and alpha_k+1
    cost = kappa(alphas, a_idx, h_bar_list)

    # check condition (25)
    print((cost - cost_prev) ** 2)
    if (cost - cost_prev) ** 2 < eps:
        break

    print("Done")

