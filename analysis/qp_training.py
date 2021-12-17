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


#### imported from analysis/test_using_towardsdatascience_lstm.py ###

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
    get_weights,
)
from functions.data_loader import load_n_filter_data
from functions.optimization_orthogonality_constraints import optimization

from ai.model_lstm import LSTMModel


file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# Variables:
batch_size = 210

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
train_data, dev_data, test_data = train_dev_test_split(
    g_recur_jets, split=[0.8, 0.1]
)

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
    h_bar_list = []

    i = 0
    for x_batch, y_batch in train_loader:
        jet_track_local = track_jets_train_data[i]
        i += 1

        x_batch = x_batch.view([batch_size, -1, model_params["input_dim"]]).to(
            device
        )
        y_batch = y_batch.to(device)

        ### Train step
        # set model to train
        lstm_model.train()

        # Makes predictions
        yhat, hn, w, r, b = lstm_model(x_batch)

        # get mean pooled hidden states
        h_bar = hn[:, jet_track_local]

        # h_bar_list.append(h_bar) # TODO, h_bar is not of fixed length! solution now: append all to list, then vstack the list to get 2 axis structure
        h_bar_list.append(h_bar)
        
        # also append w, r and b values

        # a =[hn.T[x] for x in jet_track][0][i,:].cpu().detach().numpy() selects i-th "mean pooled output"
        # a.dot(a) = h.T * h = scalar
    return torch.vstack([h_bar[0] for h_bar in h_bar_list])


def lstm_results_np(lstm_model, train_loader):
    h_bar_list = lstm_results(lstm_model, train_loader)
    return np.array([h_bar.detach().numpy() for h_bar in h_bar_list])


def kappa(alphas, a_idx, h_list):
    out = 0
    for idx1, i in enumerate(a_idx):
        for idx2, j in enumerate(a_idx):
            out += 0.5 * alphas[0, idx1] * alphas[0, idx2] * (h_list[i].T @ h_list[j])
    return out


def diff_argument(lstm_model, train_loader, alphas, a_idx, lr, W=None, R=None, b=None):
    
    h_list = lstm_results(lstm_model, train_loader)
    
    with torch.no_grad():
        if W:
            delta = W - lr * W
            lstm_model.lstm.weight_ih_l0.copy_(delta)
        elif R:
            delta = R - lr * R
            lstm_model.lstm.weight_hh_l0.copy_(delta)
        elif b:
            lstm_model.lstm.weight_ih_l0 
    
    h_list_new = lstm_results(lstm_model, train_loader)
    
    return (kappa(alphas, a_idx, h_list) - kappa(alphas, a_idx, h_list_new) ) / delta


def delta_func(lstm_model, train_loader, h_list, parm, parm_name, mu):
    
    delta = parm - mu * parm
    with torch.no_grad():
        lstm_model_new = lstm_model
        if parm_name == 'W':
            lstm_model_new.lstm.weight_ih_l0.copy_(delta)
        elif parm_name == 'R':
            lstm_model_new.lstm.weight_hh_l0.copy_(delta)
        elif parm_name == 'b':
            lstm_model_new.lstm.bias_hh_l0.copy_(delta)
    h_list_new = lstm_results(lstm_model_new, train_loader)
    return ( kappa(alphas, a_idx, h_list) - kappa(alphas, a_idx, h_list_new) ) / delta


def optimization(lstm_model, train_loader, alphas, a_idx, mu):

    # obtain W, R and b from current h
    # W = h.parameters
    # R = h.parameters
    # b = h.parameters
    #W, R, b = get_weights(lstm_model, batch_size=len())
    #dh_list = np.diff(h_list)
    #dW_list = np.diff()

    h_list = lstm_results(lstm_model, train_loader)

    # derivative of function e.g. F = (25) from Tolga
    W = lstm_model.lstm.weight_ih_l0
    G = delta_func(lstm_model, train_loader, h_list, W, 'W', mu)
    #G = np.diff(kappa(alphas, a_idx, h_list) - kappa(alphas, a_idx, h_list_new) ) / np.diff(W - mu* W)
    #G = np.gradient(lambda W: diff_argument(lstm_model, train_loader, alphas, a_idx, W=W), dW)
    A = G @ W.T - W @ G.T
    I = torch.eye(W.shape[0])
    # next point from Crank-Nicolson-like scheme
    W_next = (I + mu / 2 * A) ** (-1) @ (I - mu / 2 * A) @ W

    # same for R and b
    R = lstm_model.lstm.weight_hh_l0
    G = delta_func(lstm_model, train_loader, h_list, R, 'R', mu)
    #dR = R - mu * R
    #with torch.no_grad():
    #    lstm_model_new = lstm_model
    #    lstm_model_new.lstm.weight_hh_l0.copy_(dR)
    #G = np.gradient(lambda R: diff_argument(lstm_model, train_loader, alphas, a_idx, R=R), dW)
    #G = (kappa(alphas, a_idx, h_list) - kappa(alphas, a_idx, h_list_new) ) / dW
    A = G @ R.T - R @ G.T
    I = torch.eye(R.shape[0])
    R_next = (I + mu / 2 * A) ** (-1) @ (I - mu / 2 * A) @ R

    # TODO: two bias values: lstm_model.lstm_bias_hh_l0 and lstm_model.lstm_bias_ih_l0
    b = lstm_model.lstm_bias_hh_l0
    #G = np.gradient(lambda b: diff_argument(lstm_model, train_loader, alphas, a_idx, b=b), dW)
    G = delta_func(lstm_model, train_loader, h_list, b, 'b', mu)
    A = G @ b.T - b @ G.T
    I = torch.eye(b.shape[0])
    b_next = (I + mu / 2 * A) ** (-1) @ (I - mu / 2 * A) @ b

    return W_next, R_next, b_next


### ALGORITHM START ###
k = -1
while k < 20:  # TODO, (kappa(theta_next, alpha_next) - kappa(theta, alpha) < eps)

    # track branch number for tracking what jet_track array to use
    k += 1

    # W, R, b = get_weights(model=lstm_model, batch_size=batch_size)
    #h_bar_list = [[h_bar.detach().numpy()[0] for h_bar in h_bar_list]]
    h_bar_list_numpied = lstm_results_np(lstm_model, train_loader)
    svm_model.fit(h_bar_list_numpied)
    alphas = np.abs(svm_model.dual_coef_)
    a_idx = svm_model.support_

    W_next, R_next, b_next = optimization(lstm_model, train_loader, alphas, a_idx, learning_rate)
    

    """
    # get rho?
    for i in range(h_bar_list):
        rho = 0
        for j in range(h_bar_list):
            rho += alpha[j] * alpha[i] * h_bar_list[j] * h_bar_list[i]
    """

