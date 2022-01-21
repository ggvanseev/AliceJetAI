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
)
from functions.data_loader import load_n_filter_data
from functions.optimization_orthogonality_constraints import (
    lstm_results,
    kappa,
    optimization,
)

from ai.model_lstm import LSTMModel

# from autograd import elementwise_grad as egrad

from copy import copy

import matplotlib.pyplot as plt

from hyperopt import (
    fmin,
    tpe,
    hp,
    space_eval,
    STATUS_OK,
    Trials,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
from functools import partial

# Set hyper space and variables

max_evals = 100
space = hp.choice(
    "hyper_parameters",
    [
        {
            "batch_size": hp.choice("num_batch", [50, 100, 200]),
            "num_epochs": hp.choice("num_epochs", [int(500)]),
            "num_layers": hp.choice("num_layers", [1]),
            "learning_rate": hp.choice("learning_rate", [1e-7, 1e-5, 0.1]),
            "decay_factor": hp.choice("decay_factor", [0.9]),
            "dropout": hp.choice("dropout", [0]),
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice(
                "svm_nu", [0.05, 0.01, 0.04, 0.001]
            ),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale", "auto"]
            ),  # , "scale", [ 0.23 was the defeault before]
        }
    ],
)

# define another space just to test a single value in this algorithm TODO remove later
space_single = hp.choice(
    "hyper_parameters",
    [
        {
            "batch_size": hp.choice("num_batch", [50]),
            "num_epochs": hp.choice("num_epochs", [int(500)]),
            "num_layers": hp.choice("num_layers", [1]),
            "learning_rate": hp.choice("learning_rate", [0.1]),
            "decay_factor": hp.choice("decay_factor", [0.9]),
            "dropout": hp.choice("dropout", [0]),
            "output_dim": hp.choice("output_dim", [3]),
            "svm_nu": hp.choice("svm_nu", [0.01]),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale", "auto"]
            ),  # , "scale", [ 0.23 was the defeault before]
        }
    ],
)


file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)

# only use g_recur_jets
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])


def try_hyperparameters(hyper_parameters: dict, train_data, dev_data):
    # Track time
    time_track = time.time()

    # Variables:
    batch_size = hyper_parameters["batch_size"]
    output_dim = hyper_parameters["output_dim"]
    layer_dim = hyper_parameters["num_layers"]
    dropout = hyper_parameters["dropout"]
    epochs = hyper_parameters["num_epochs"]
    learning_rate = hyper_parameters["learning_rate"]
    svm_nu = hyper_parameters["svm_nu"]
    svm_gamma = hyper_parameters["svm_gamma"]
    # weight_decay = 1e-6
    eps = 1e-2  # test value for convergence

    # Show used hyper_parameters in terminal
    print("Hyper Parameters")
    print(hyper_parameters)

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
    svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="rbf")

    # path for model - only used for saving
    # model_path = f'models/{lstm_model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ### ALGORITHM START ###
    """
    alphas = np.abs(svm_model.dual_coef_)
    a_idx = svm_model.support_"""
    # calculate cost function kappa with alpha_0 and theta_0:
    # TODO set alphas to 1 or 0 so cost_next - cost will be large
    cost = 1e10  # kappa(alphas, a_idx, h_bar_list)
    cost_prev = 1e9

    # obtain h_bar from the lstm with theta_0, given the data
    h_bar_list, theta, theta_gradients = lstm_results(
        lstm_model,
        model_params,
        train_loader,
        track_jets_train_data,
        batch_size,
        device,
    )
    h_bar_list_np = np.array([h_bar.detach().numpy() for h_bar in h_bar_list])

    # Array to track cost
    # track_cost = np.zeros(epochs + 1)
    # track_cost_condition = np.zeros(epochs + 1)
    track_cost = []  # TODO
    track_cost_condition = []

    k = -1
    while (
        # k < epochs
        (cost - cost_prev) ** 2
        >= eps
    ):
        k += 1

        # keep previous cost result stored
        cost_prev = copy(cost)

        # obtain alpha_k+1 from the h_bars with SMO through the OC-SVMs .fit()
        svm_model.fit(h_bar_list_np)
        alphas = np.abs(svm_model.dual_coef_)[0]

        a_idx = svm_model.support_

        # obtain theta_k+1 using the optimization algorithm
        lstm_model, theta_next = optimization(
            lstm=lstm_model,
            alphas=alphas,
            a_idx=a_idx,
            mu=learning_rate,
            h_bar_list=h_bar_list,
            theta=theta,
            theta_gradients=theta_gradients,
            device=device,
        )

        # obtain h_bar from the lstm with theta_k+1, given the data
        h_bar_list, theta, theta_gradients = lstm_results(
            lstm_model,
            model_params,
            train_loader,
            track_jets_train_data,
            batch_size,
            device,
        )
        h_bar_list_np = np.array([h_bar.detach().numpy() for h_bar in h_bar_list])

        # obtain the new cost given theta_k+1 and alpha_k+1
        cost = kappa(alphas, a_idx, h_bar_list)

        # check condition (25)
        # print((cost - cost_prev) ** 2)
        # track_cost[k] = cost
        # track_cost_condition[k] = (cost - cost_prev) ** 2
        track_cost.append(cost)
        track_cost_condition.append((cost - cost_prev) ** 2)

        if np.isnan(track_cost_condition[k]):
            print("Broke, for given hyper parameters")
            return 1e10  # Return large number to not be selected as the best

        if (cost - cost_prev) ** 2 < eps:
            print("Succes, epoch {}".format(k), cost, cost_prev, sep="\n")
            break

    print(f"Done in: {time.time()-time_track}")

    # plot cost condition and cost function
    title_plot = f"plot_with_{epochs}_epochs_{batch_size}_batch_size_{learning_rate}_learning_rate_{svm_gamma}_svm_gamma_{svm_nu}_svm_nu"
    fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
    fig.suptitle(title_plot, y=1.08)
    ax1.plot(track_cost_condition[1:])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cost Condition")

    ax2 = ax1.twinx()
    ax2.plot(track_cost[1:], "--", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Cost")

    fig.savefig("output/" + title_plot + ".png")

    # To fulfill codition using fmin return similair as
    # TODO return the cost of last epoch, where the objective function has converged?
    return {"loss": np.min(track_cost_condition), "status": STATUS_OK}


best = fmin(
    partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
        try_hyperparameters, train_data=train_data, dev_data=dev_data
    ),
    space,
    algo=tpe.suggest,
    max_evals=30,  # max_evals, TODO
)

