from typing import no_type_check

import torch

import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

from functions.data_saver import save_results, save_loss_plots, DataTrackerTrials

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials

import names as na
from ai.model import LSTM
from functions import data_loader


# load root dataset into a pandas dataframe
fileName = "./samples/JetToyHIResultSoftDropSkinny.root"
g_jets, q_jets, g_recur_jets, q_recur_jets = data_loader.load_n_filter_data(
    fileName, na.tree
)

# weighted mse loss
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)


# weighted bce loss
def weighted_bce_loss(input, target, weight):
    return torch.nn.functional.binary_cross_entropy(
        input, target, weight, reduction="sum"
    ) / torch.sum(weight)


def get_loss_training(loss_func, out, label, weight):
    res = nn.functional.softmax(out, dim=1)
    if loss_func == "mse":
        loss = weighted_mse_loss(res, label, weight)
    elif loss_func == "bce":
        loss = weighted_bce_loss(res, label, weight)

    return loss


def train_model(
    training_data,
    model,
    device,
    num_epochs,
    learning_rate,
    decay_factor,
    loss_func="mse",
):
    """
    training of the model with a give num_epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate decay exponentially
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, decay_factor, last_epoch=-1
    )

    # training
    step_training = []
    loss_training = []

    # Epoch is the number of times, all training data is taken through
    for epoch in range(num_epochs):

        scheduler.step()

        # enumerate over all available data
        for step, (seq, weight, label, length) in enumerate(training_data):

            # train the model for a given (data-)sequence
            seq = seq.to(
                device
            )  # Make data sequence match desired device model is running on
            out = model(seq)  # Get ouput
            out = out.to(torch.device("cpu"))

            # Get loss from training
            loss = get_loss_training(loss_func, out, label, weight)

            # Optimize training (?)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Store training data
            step_training.append(step + 1)
            loss_training.append(float(loss))

            # Track progres training in terminal
            if (step + 1) % 10 == 0:
                print(
                    "Eopch: [{}/{}], LR: {}, Step: {}, Loss:{:.4f}".format(
                        epoch,
                        num_epochs,
                        scheduler.get_lr(),
                        step + 1,
                        float(loss),
                    )
                )

    return model, step_training, loss_training


def validate_model(model, device, validation_data, loss_func):
    loss_sum = 0.0
    weight_sum = 0.0

    # Trun moddel into evaluation mode
    model.eval()
    for step, (seq, weight, label, length) in enumerate(validation_data):
        seq = seq.to(device)
        # Use torch.no_grad since it is validation (?)
        with torch.no_grad():
            # Get output
            out = model(seq)
            out = out.to(torch.device("cpu"))

            # Get sum of loss (note different from get loss from training)
            res = nn.functional.softmax(out, dim=1)
            if loss_func == "bce":
                loss_sum += float(
                    torch.nn.functional.binary_cross_entropy(
                        res, label, weight, reduction="sum"
                    )
                )
            elif loss_func == "mse":
                loss_sum += float(torch.sum(weight * (res - label) ** 2))

            # get sum of weight
            weight_sum += torch.sum(weight)

    # Return result of validation
    return loss_sum / weight_sum


def train_model_with_hyper_parameters(
    hyper_parameters: dict,
    training_data: ak,
    validation_data: ak,
    data_tracker: DataTrackerTrials,
    model=LSTM,
    n_attempts: int = 3,
    flag_save_loss_plots: bool = False,
    flag_save_intermediate_results: bool = False,
    jetpt_cut=130,
    loss_minimum=2,
    model_best=None,
):

    # Get hyper parameter settings
    num_batch = int(hyper_parameters["num_batch"])
    num_epochs = int(hyper_parameters["num_epochs"])
    num_layers = int(hyper_parameters["num_layers"])
    hidden_size0 = int(hyper_parameters["hidden_size0"])
    hidden_size1 = int(hyper_parameters["hidden_size1"])
    learning_rate = hyper_parameters["learning_rate"]
    decay_factor = hyper_parameters["decay_factor"]
    loss_func = hyper_parameters["loss_func"]

    # Show used hyper_parameters in terminal
    print("Hyper Parameters")
    print(hyper_parameters)

    # Get machine type:
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    # Prepare training and valuation data for Model use
    training_data = data.DataLoader(
        training_data,
        batch_size=num_batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn_pad,
    )

    validation_data = data.DataLoader(
        validation_data,
        batch_size=num_batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn_pad,
    )

    # Try to train moddel:
    for i in range(n_attempts):
        # Track attempt
        print(f"Trial: {data_tracker.i_trial}. Attempt: {i}.\n")

        # Declare model
        model = model(
            input_size=4,
            hidden_size=[hidden_size0, hidden_size1],
            num_layers=num_layers,
            batch_size=num_batch,
            device=device,
        )

        model, step_training, loss_training = train_model(
            training_data,
            model,
            device,
            num_epochs,
            learning_rate,
            decay_factor,
            loss_func,
        )

    # If flag is true, save loss plots
    if flag_save_loss_plots:
        save_loss_plots(
            step_training, loss_training, data_tracker.i_trial, i, loss_func
        )

    if flag_save_intermediate_results:
        prefix = (
            "zcut0p1_beta0_csejet_"
            + loss_func
            + "_mult7000_pt"
            + str(jetpt_cut)
            + "_itrial_"
            + str(data_tracker.i_trial)
            + "_"
            + str(i)
        )
        save_results(prefix, model, hyper_parameters)

    # validation
    result = validate_model(model, device, validation_data, loss_func)

    # Return model to training mode after validating
    model.train()

    # Check if trial is lowest loss
    if result.item() < loss_minimum:
        loss_minimum = result.item()
        model_best = model

    print(
        "Trial: {}, Attempt: {}, Validation Loss {}".format(
            data_tracker.i_trial, i, result.item()
        )
    )

    # Save results of trial
    data_tracker.index_trial.append(data_tracker.i_trial)
    data_tracker.loss_trial.append(loss_minimum)

    if loss_func == "bce":
        print("Validation (BCE LOSS): %.4f" % loss_minimum)
    elif loss_func == "mse":
        print("Validation (MSE LOSS): %.4f" % loss_minimum)

    return {
        "loss": loss_minimum,
        "status": STATUS_OK,
        "model": model_best,
        "loss_func": loss_func,
    }


def getBestModelfromTrials(trials):
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial["result"]["status"]
    ]
    losses = [float(trial["result"]["loss"]) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj["result"]["model"], best_trial_obj["result"]["loss_func"]
