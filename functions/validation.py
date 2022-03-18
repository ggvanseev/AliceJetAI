import torch
import numpy as np
from functions.data_manipulation import h_bar_list_to_numpy

from functions.run_lstm import calc_lstm_results


def calc_percentage_anomalies(
    data_loader,
    track_jets_data,
    input_dim,
    lstm_model,
    svm_model,
    pooling="last",
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    """
    Returns the absolute difference between the chosen nu
    """

    h_bar_list, _, _ = calc_lstm_results(
        lstm_model,
        input_dim,
        data_loader,
        track_jets_data,
        device,
        pooling,
    )

    h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

    # get prediction
    h_predicted = svm_model.predict(h_bar_list_np)

    # count anomalies
    n_anomaly = np.count_nonzero(h_predicted == -1)

    fraction_anomaly = n_anomaly / len(h_predicted)

    return fraction_anomaly


def validation_distance_nu(
    nu,
    val_loader,
    track_jets_val_data,
    input_dim,
    lstm_model,
    svm_model,
    pooling="last",
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    """
    Returns the absolute difference between the chosen nu
    """

    fraction_anomaly = calc_percentage_anomalies(
        val_loader,
        track_jets_val_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device=device,
    )

    return abs(fraction_anomaly - nu)
