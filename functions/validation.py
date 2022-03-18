import torch
import numpy as np
from functions.data_manipulation import h_bar_list_to_numpy


def calc_percentage_anomalies(
    val_loader,
    track_jets_val_data,
    input_dim,
    lstm_model,
    svm_model,
    pooling,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    """
    Returns the absolute difference between the chosen nu
    """

    h_bar_list = []

    with torch.no_grad():
        i = 0
        for x_batch, y_batch in val_loader:
            jet_track_local = track_jets_val_data[i]
            i += 1

            x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(device)
            y_batch = y_batch.to(device)

            ### Train step
            # set model to train

            # Makes predictions, and don't use backpropagation
            hn = lstm_model(x_batch, backpropagation_flag=False)

            # get mean pooled hidden states
            if pooling == "last":
                h_bar = hn[:, jet_track_local]
            elif pooling == "mean":
                h_bar = torch.zeros([1, len(jet_track_local), hn.shape[-1]])
                jet_track_prev = 0
                jet_track_local = [x + 1 for x in jet_track_local]
                jet_track_local[-1] = None
                for j, jet_track in enumerate(jet_track_local):
                    h_bar[:, j] = torch.mean(hn[:, jet_track_prev:jet_track], dim=1)
                    jet_track_prev = jet_track

            h_bar_list.append(h_bar)

    # Take last layer
    h_bar_list = torch.vstack([h_bar[-1] for h_bar in h_bar_list])

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
