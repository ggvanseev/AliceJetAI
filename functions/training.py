from typing import no_type_check

import torch
from torch.utils import data
import numpy as np

import awkward as ak

from functions.data_saver import save_results, save_loss_plots, DataTrackerTrials
from functions.data_manipulation import collate_fn_pad

from hyperopt import STATUS_OK

from ai.model import LSTM_FC


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
    return None


def validate_model(model, device, validation_data, loss_func):
    """
    Validate model by making ROC curve
    """
    # Trun moddel into evaluation mode
    model.eval()
    return None


def train_model_with_hyper_parameters(
    hyper_parameters: dict,
    training_data: ak,
    validation_data: ak,
    data_tracker: DataTrackerTrials,
    output_dim: int,
    model=LSTM_FC,
    n_attempts: int = 3,
    flag_save_intermediate_results: bool = False,
    jetpt_cut=130,
    loss_minimum=2,
    model_best=None,
):
    """
    The function is ready to train for different hyper_parameters using fmin and Train from hyperoot
    library.For an example see training_an_lstm.py.
    In addition it can save (figures of) intermdiate results.
    Variables:
    hyper_parameters: dict of hyper parameters, see needed once at begening of this function,
    training_data: class with the training data,
    validation_data: class with the validation data,
    data_tracker: class of type DataTrackerTrials to track progress without returning these values.
    This is needed due to the inner working of fmin, which only allows a specified output to come from
    the function.
    output_dim: int, with the dimension of the (1D) output.
    model=LSTM_FC, model to use, standard is LSTM_FC. Note: if using a different model, make sure the
    model has the same parameters as used in this function.
    n_attempts: int, number of attempts on trying to train a model.
    flag_save_intermediate_results: bool, see reference above.
    jetpt_cut=130, minium jet energy. Note, use the same as in selection of train/validation data
    loss_minimum=2, minum loss per training
    model_best=None, track best model, in the beging None.

    """

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    # Prepare training and valuation data for Model use using torch library provided DataLoader
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
            output_dim=output_dim,
            num_layers=num_layers,
            batch_size=num_batch,
            device=torch.device("cpu"),
        )

        # Train model
        model, step_training, loss_training = train_model(
            training_data,
            model,
            device,
            num_epochs,
            learning_rate,
            decay_factor,
            loss_func,
        )

    # If flag is true, save intermediate results
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

    # Check if trial is lowest loss, if so lower minum loss and update best model
    if result.item() < loss_minimum:
        loss_minimum = result.item()
        model_best = model

    # Track results
    print(
        "Trial: {}, Attempt: {}, Validation Loss {}".format(
            data_tracker.i_trial, i, result.item()
        )
    )

    # Save results of trial
    data_tracker.index_trial.append(data_tracker.i_trial)
    data_tracker.loss_trial.append(loss_minimum)

    return {
        "criterium of succes": loss_minimum,  # criterium which determines succes of model
        "status": STATUS_OK,
        "model": model_best,
        "loss_func": loss_func,
    }


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
