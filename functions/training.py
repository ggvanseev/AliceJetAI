import torch
import numpy as np

from hyperopt import STATUS_OK

import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib
# import joblib

import time

import torch

from functions.data_saver import save_results
from functions.data_manipulation import (
    branch_filler,
    lstm_data_prep,
    h_bar_list_to_numpy,
)
from functions.optimization_orthogonality_constraints import (
    lstm_results,
    kappa,
    optimization,
)

from functions.validation import validation_distance_nu

from ai.model_lstm import LSTMModel

# from autograd import elementwise_grad as egrad

from copy import copy

import matplotlib.pyplot as plt


def getBestModelfromTrials(trials):
    """
    Find best model from trials, using the hypertool training
    """
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial["result"]["status"]
    ]

    # Replace losses, by crtiterium for future best model determination
    losses = [float(trial["result"]["loss"]) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]

    return best_trial_obj["result"]["model"], best_trial_obj["result"]["loss_func"]


def training_algorithm(
    lstm_model,
    svm_model,
    x_loader,
    track_jets_dev_data,
    model_params,
    training_params,
    device,
):
    """
    Trainging algorithm 1 from paper Tolga: Unsupervised Anomaly Detection With LSTM Neural Networks
    """
    # path for model - only used for saving
    # model_path = f'models/{lstm_model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    ### ALGORITHM START ###
    # Set initial cost
    # TODO set alphas to 1 or 0 so cost_next - cost will be large
    cost = 1e10
    cost_condition_passed_flag = (
        False  # flag to check if cost condition has been satisfied
    )
    min_epochs_patience = training_params["min_epochs"]

    # obtain h_bar from the lstm with theta_0, given the data
    h_bar_list, theta, theta_gradients = lstm_results(
        lstm_model,
        model_params["input_dim"],
        x_loader,
        track_jets_dev_data,
        device,
    )
    h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

    # list to track cost
    track_cost = []
    track_cost_condition = []

    # loop over k (epochs) for nr. set epochs and unsatisfied cost condition
    k = -1
    while (
        k < min_epochs_patience or cost_condition_passed_flag == False
    ) and k < training_params["max_epochs"]:
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
            mu=training_params["learning_rate"],
            h_bar_list=h_bar_list,
            theta=theta,
            theta_gradients=theta_gradients,
            device=device,
        )

        # obtain h_bar from the lstm with theta_k+1, given the data
        h_bar_list, theta, theta_gradients = lstm_results(
            lstm_model,
            model_params["input_dim"],
            x_loader,
            track_jets_dev_data,
            device,
        )
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

        # obtain the new cost and cost condition given theta_k+1 and alpha_k+1
        cost = kappa(alphas, a_idx, h_bar_list)
        cost_condition = (cost - cost_prev) ** 2

        # track cost and cost_condition
        track_cost.append(cost)
        track_cost_condition.append(cost_condition)

        # check condition algorithm 1, paper Tolga
        if (cost - cost_prev) ** 2 < training_params["epsilon"]:
            # check if condition had been satisfied recently
            if cost_condition_passed_flag == False:
                cost_condition_passed_flag = True
                # check if k + patience would be larger than minimum number of epochs
                # Update min_epochs_patience to check if
                if k + training_params["patience"] > training_params["min_epochs"]:
                    min_epochs_patience = k + training_params["patience"]
        else:
            cost_condition_passed_flag = False

        # Check if cost function starts to explode
        if np.isnan(track_cost_condition[k]):
            print("Broke, for given hyper parameters")
            return 1e10  # Return large number to not be selected as the best

    if (cost - cost_prev) ** 2 > training_params["epsilon"]:
        print("Alogirthm failed: not done learning in max epochs.")
        passed = False
    else:
        print(f"Model done learning in {k} epochs.")
        passed = True

    return lstm_model, svm_model, track_cost, track_cost_condition, passed


def try_hyperparameters(
    hyper_parameters: dict,
    dev_data,
    val_data,
    plot_flag: bool = False,
    max_epochs=5000,
    eps=1e-6,
    patience=50,
    max_attempts=4,
    max_distance_nu=0.01,
):
    """
    This function searches for the correct hyperparameters.

    It follows the following procedure:
    1. Assign given variables for a run from hyper_parameters.
    2. Prepare data with given parameters.
    3. Try to find distance nu and errors in dev_data prediction < 0.01:
        3.1. Run algorithm 1 from paper tolga..
        Use condition from algorithm to see if the model is still learning.
    4. Save and plot if flags are true
    5. Return distance nu and dev_data as loss.

    """
    # Track time
    time_track = time.time()

    # use correct device:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Variables:
    batch_size = hyper_parameters["batch_size"]
    output_dim = hyper_parameters["output_dim"]
    layer_dim = hyper_parameters["num_layers"]
    dropout = hyper_parameters["dropout"]
    min_epochs = hyper_parameters["min_epochs"]
    learning_rate = hyper_parameters["learning_rate"]
    svm_nu = hyper_parameters["svm_nu"]
    svm_gamma = hyper_parameters["svm_gamma"]
    hidden_dim = hyper_parameters["hidden_dim"]

    # Set episilon to be 100 times smaller than the learning factor:
    eps = learning_rate * 1e-2

    # Show used hyper_parameters in terminal
    print("Hyper Parameters")
    print(hyper_parameters)

    # prepare data for usage
    dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)
    val_data, track_jets_val_data = branch_filler(val_data, batch_size=batch_size)

    # Only use train and dev data for now
    # Note this has to be saved with the model, to ensure data has the same form.
    scaler = MinMaxScaler()
    dev_loader = lstm_data_prep(
        data=dev_data, scaler=scaler, batch_size=batch_size, fit_flag=True
    )
    val_loader = lstm_data_prep(data=val_data, scaler=scaler, batch_size=batch_size)

    # set model parameters
    input_dim = len(dev_data[0])
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "dropout_prob": dropout,
    }

    # set training parameters
    training_params = {
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "epsilon": eps,
        "patience": patience,
    }

    n_attempt = 0
    while n_attempt < max_attempts:
        n_attempt += 1

        # Declare models
        lstm_model = LSTMModel(**model_params)
        svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="linear")

        # set model to correct device
        lstm_model.to(device)

        (
            lstm_model,
            svm_model,
            track_cost,
            track_cost_condition,
            passed,
        ) = training_algorithm(
            lstm_model,
            svm_model,
            dev_loader,
            track_jets_dev_data,
            model_params,
            training_params,
            device,
        )

        # Check if distance to svm_nu is smaller than required

        distance_nu = validation_distance_nu(
            svm_nu,
            dev_loader,
            track_jets_dev_data,
            input_dim,
            lstm_model,
            svm_model,
            device,
        )

        if distance_nu < max_distance_nu and track_cost[0] != track_cost[-1] and passed:
            n_attempt = max_attempts
        else:
            distance_nu = (
                10  # Add large distance to ensure wrong model doesn't end up in list
            )
            plot_flag = False

    print(f"Done in: {time.time()-time_track}")

    if plot_flag:
        # plot cost condition and cost function
        title_plot = f"plot_with_{max_epochs}_epochs_{batch_size}_batch_size_{learning_rate}_learning_rate_{svm_gamma}_svm_gamma_{svm_nu}_svm_nu"
        fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
        fig.suptitle(title_plot, y=1.08)
        ax1.plot(track_cost_condition[1:])
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Cost Condition")

        ax2 = ax1.twinx()
        ax2.plot(track_cost[1:], "--", linewidth=0.5, alpha=0.7)
        ax2.set_ylabel("Cost")

        fig.savefig("output/" + title_plot + str(time.time()) + ".png")

    # return the model
    lstm_ocsvm = dict({"lstm:": lstm_model, "ocsvm": svm_model, "scaler": scaler})

    return {"loss": distance_nu, "status": STATUS_OK, "model": lstm_ocsvm}
