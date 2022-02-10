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
    scaled_epsilon_n_max_epochs,
    branch_filler_jit,
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

    # Replace losses, by crtiterium for future best model determination TODO
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

    ### TRACK TIME ### TODO
    time_at_step = time.time()

    # Set initial cost
    # TODO set alphas to 1 or 0 so cost_next - cost will be large
    cost = 1e10
    cost_condition_passed_flag = (
        False  # flag to check if cost condition has been satisfied
    )
    min_epochs_patience = training_params["min_epochs"]

    # obtain h_bar from the lstm with theta_0, given the data
    h_bar_list, theta, theta_gradients = lstm_results(
        lstm_model, model_params["input_dim"], x_loader, track_jets_dev_data, device,
    )
    h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

    # list to track cost
    track_cost = []
    track_cost_condition = []

    ### TRACK TIME ### TODO
    # dt = time.time() - time_at_step
    # time_at_step = time.time()
    # print(f"Obtained first h_bars, done in: {dt}")

    # loop over k (epochs) for nr. set epochs and unsatisfied cost condition
    k = -1
    while (
        k < min_epochs_patience or cost_condition_passed_flag == False
    ) and k < training_params["max_epochs"]:
        k += 1

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Start of loop, done in: {dt} \t epoch {k}")

        # keep previous cost result stored
        cost_prev = copy(cost)

        # obtain alpha_k+1 from the h_bars with SMO through the OC-SVMs .fit()
        svm_model.fit(h_bar_list_np)
        alphas = np.abs(svm_model.dual_coef_)[0]

        a_idx = svm_model.support_

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained alphas, done in: {dt}")

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

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained thetas, done in: {dt}")

        # obtain h_bar from the lstm with theta_k+1, given the data
        h_bar_list, theta, theta_gradients = lstm_results(
            lstm_model,
            model_params["input_dim"],
            x_loader,
            track_jets_dev_data,
            device,
        )
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained h_bar, done in: {dt}")

        # obtain the new cost and cost condition given theta_k+1 and alpha_k+1
        cost = kappa(alphas, a_idx, h_bar_list)
        cost_condition = (cost - cost_prev) ** 2

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained cost, done in: {dt}")

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
            return (
                lstm_model,
                svm_model,
                track_cost,
                track_cost_condition,
                False,
            )  # immediately return passed = False

    if (cost - cost_prev) ** 2 > training_params["epsilon"]:
        print("Algorithm failed: not done learning in max epochs.")
        passed = False
    else:
        print(f"Model done learning in {k} epochs.")
        passed = True

    return lstm_model, svm_model, track_cost, track_cost_condition, passed


def try_hyperparameters(
    hyper_parameters: dict,
    dev_data,
    plot_flag: bool = False,
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

    # Variables:
    batch_size = int(hyper_parameters["batch_size"])
    output_dim = int(hyper_parameters["output_dim"])
    layer_dim = int(hyper_parameters["num_layers"])
    dropout = hyper_parameters["dropout"] if layer_dim > 1 else 0  # TODO
    min_epochs = hyper_parameters["min_epochs"]
    learning_rate = hyper_parameters["learning_rate"]
    svm_nu = hyper_parameters["svm_nu"]
    svm_gamma = hyper_parameters["svm_gamma"]
    hidden_dim = int(hyper_parameters["hidden_dim"])

    # Set epsilon and max_epochs
    eps, max_epochs = scaled_epsilon_n_max_epochs(learning_rate)

    # Show used hyper_parameters in terminal
    # sauce https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
    print("\nHyper Parameters:")
    print("\n".join("  {:10}\t  {}".format(k, v) for k, v in hyper_parameters.items()))

    # use correct device:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: {}".format(device))

    # prepare data for usage
    # dev_data_copy = copy(dev_data)  # save this to check the error of data[] TODO
    # try:
    #     dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)
    # except TypeError:
    #     print("Could not create jet branch with given data and parameters!")
    #     return (
    #         10  # for "loss", since this will be added to the 1st column of the result
    #     )

    time_track = time.time()
    dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)
    dt = time.time() - time_track
    print(f"Branch filler jit, done in: {dt}")

    # Only use train and dev data for now
    # Note this has to be saved with the model, to ensure data has the same form.
    scaler = MinMaxScaler()
    dev_loader = lstm_data_prep(
        data=dev_data, scaler=scaler, batch_size=batch_size, fit_flag=True
    )

    # set model parameters
    input_dim = len(dev_data[0])
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "dropout_prob": dropout,
        "batch_size": batch_size,
        "device": device,
    }

    # set training parameters
    training_params = {
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "epsilon": eps,
        "patience": patience,
    }

    ### TRACK TIME ### TODO
    dt = time.time() - time_track
    print(f"Dataprep, done in: {dt}")

    n_attempt = 0
    while n_attempt < max_attempts:
        n_attempt += 1

        # Declare models
        lstm_model = LSTMModel(**model_params)
        svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="linear")

        # set model to correct device
        lstm_model.to(device)

        try:
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
        except RuntimeError as e:
            passed = False
            logf = open("logfiles/cuda_error.log", "w")
            logf.write(str(e))

        # check if the model passed the training
        distance_nu = (
            10  # Add large distance to ensure wrong model doesn't end up in list
        )
        train_success = False
        if passed:
            distance_nu = validation_distance_nu(
                svm_nu,
                dev_loader,
                track_jets_dev_data,
                input_dim,
                lstm_model,
                svm_model,
                device,
            )

            # Check if distance to svm_nu is smaller than required
            if distance_nu < max_distance_nu and track_cost[0] != track_cost[-1]:
                n_attempt = max_attempts
                train_success = True

    print(f"{'Passed' if train_success else 'Failed'} in: {time.time()-time_track}")

    if plot_flag:
        # plot cost condition and cost function
        title_plot = f"plot_with_{max_epochs}_epochs_{batch_size}_batch_size_{learning_rate}_learning_rate_{svm_gamma}_svm_gamma_{svm_nu}_svm_nu_{distance_nu}_distance_nu"
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

    # save plot data
    cost_data = dict(
        {"cost": track_cost[1:], "cost_condition": track_cost_condition[1:]}
    )

    return {
        "loss": distance_nu,
        "final_cost": track_cost[-1],
        "status": STATUS_OK,  # update with train_success? TODO
        "model": lstm_ocsvm,
        "hyper_parameters": hyper_parameters,
        "cost_data": cost_data,
        "num_batches": len(dev_loader),
    }
