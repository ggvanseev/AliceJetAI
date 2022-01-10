import torch
from copy import copy

from functions.data_manipulation import (
    get_full_pytorch_weight,
    put_weight_in_pytorch_matrix,
)

import time
import numpy as np


def lstm_results(
    lstm_model,
    model_params,
    train_loader,
    track_jets_train_data,
    batch_size,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    """Obtain h_bar states from the lstm with the data

    Args:
        lstm_model (LSTMModel): the LSTM model
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch, stores
                the data
        model_params ():
        track_jets_train_data ():
        batch_size ():
        device ():

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
        lstm_model.train()  # TODO should this be off so the backward() call in the forward pass does not update the weights?

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

    # Get mean of theta gradients
    for key1, value in theta_gradients_temp.items():
        for key2, value2 in value.items():
            theta_gradients[key1][key2] = theta_gradients[key1][key2] / len(
                train_loader
            )

    return torch.vstack([h_bar[0] for h_bar in h_bar_list]), theta, theta_gradients


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
    # Use torch.no_grad to not record changes in this section
    with torch.no_grad():
        h_matrix = (
            h_list[a_idx] @ h_list[a_idx].T
        )  # Matrix multiplication is time consuming, thus to do this as least as possbile do this
        # once and create a matrix with the results

        # out1 = 0
        # for idx1, i in enumerate(a_idx):
        #     for idx2, j in enumerate(a_idx):
        #         out1 += (
        #             0.5 * alphas[idx1] * alphas[idx2] * h_matrix[idx1, idx2]
        #         )  # (h_list[i].T @ h_list[j])

        out = 0  # use for trackking summation
        n_alphas = len(a_idx)

        alphas_matrix = np.outer(alphas, alphas).T

        for i in range(n_alphas):
            # Slight difference compared to taking for loops (probably due to numerical solution in numpy), but this is a systematic error that cancels out (?) when comparing two kappas.
            out += 0.5 * np.dot(alphas_matrix[i], h_matrix[i])

    return out


def delta_func(
    lstm_model,
    model_params,
    train_loader,
    track_jets_train_data,
    batch_size,
    h_list,
    theta_gradients,
    weight,
    weight_name: str,
    mu,
    alphas,
    a_idx,
    pytorch_weights,
    device,
):
    """Calculates the derivative of G to a specific weight or bias.
    G = dkappa / dW_ij = alpha_i * alpha_j * h_ij * dh_ij / dW_ij
    since the derivative of x^T x = 2x
    dh / dW can be obtained from theta_gradients

    d(h_i.T * h_j) / dh =  ((dh_i / dW) * h_j + (h_i.T dh_j/dW))
    = (h_i + h_j) * dh/dW


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

    # d_weight = mu * weight
    # new_weight = weight - d_weight

    # # Use torch.no_grad to not record changes in this section
    # with torch.no_grad():

    #     lstm_model_new = copy(
    #         lstm_model  # Needs a copy, to avoid unexpected changes in the original model
    #     )

    #     # only updated desired weight element
    #     pytorch_weights = put_weight_in_pytorch_matrix(
    #         new_weight, weight_name, pytorch_weights
    #     )

    #     getattr(lstm_model_new.lstm, weight_name[5:]).copy_(pytorch_weights)

    # h_list_new, _ = lstm_results(
    #     lstm_model_new,
    #     model_params,
    #     train_loader,
    #     track_jets_train_data,
    #     batch_size,
    #     device,
    # )
    # return (kappa(alphas, a_idx, h_list_new) - kappa(alphas, a_idx, h_list)) / (
    #     new_weight - weight
    # )
    # Use torch.no_grad to not record changes in this section
    with torch.no_grad():
        out = 0
        for idx1, i in enumerate(a_idx):
            for idx2, j in enumerate(a_idx):
                out += (
                    alphas[0, idx1]
                    * alphas[0, idx2]
                    * (h_list[i] + h_list[j])
                    @ theta_gradients
                )
    return out


def calc_g(gradient_hi, h_bar_list, alphas, a_idx):
    """Calculates the derivative of G to a specific weight or bias.
    G = dkappa / dW_ij = (dkappa * dh_ij) *(dh_ij / dW_ij)
    Since the derivative of x^T x = 2x and
    dh / dW can be obtained from theta_gradients
    this is

    G = dkappa / dW_ij = (dkappa * dh_ij) *(dh_ij / dW_ij) =
    (0.5*sumi,j alpah_i*alpha_j*2*hi) * dh/dw(theta_gradients)
    = sum_alhpa_j*alpha_i*h_i

    proof: https://math.stackexchange.com/questions/1377764/derivative-of-vector-and-vector-transpose-product
    """

    # TODO: the looping takes to long (8 seconds with this small batch, maybe there is a way to speed it up)
    alphas_j = np.sum(alphas)

    d_kappa = (
        alphas_j * torch.tensor(alphas).type(torch.FloatTensor) @ h_bar_list[a_idx]
    )

    return (
        d_kappa * gradient_hi.T
    ).T  # TODO: check if this the correct way of multiplication


def updating_theta(
    h_bar_list,
    theta: dict,
    theta_gradients: dict,
    mu,
    alphas,
    a_idx,
    device,
):
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

        gradient_weights = theta_gradients[weight_type]

        for weight_name, weight in weights.items():
            # follow steps from eq. 24 in paper Tolga

            gradient_weight = gradient_weights[weight_name]

            # derivative of function e.g. F = (25) from Tolga
            g = calc_g(gradient_weight, h_bar_list, alphas, a_idx)

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

        # Get weight name without layer number
        weight_name = list(weights.keys())[0][5:-1]

        # Loop over all layers
        for i in range(len(weights) // 4):
            with torch.no_grad():
                getattr(lstm.lstm, weight_name + str(i)).copy_(pytorch_weights[str(i)])

    return lstm


def optimization(
    lstm,
    alphas,
    a_idx,
    mu,
    h_bar_list,
    theta,
    theta_gradients,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
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
    theta = updating_theta(
        h_bar_list,
        theta,
        theta_gradients,
        mu,
        alphas,
        a_idx,
        device,
    )

    # update lstm
    lstm = update_lstm(lstm, theta)

    return lstm, theta


# def optimization(model, h_list, alphas, a_idx, mu):

#     # obtain W, R and b from current h
#     # W = h.parameters
#     # R = h.parameters
#     # b = h.parameters
#     W, R, b = get_weights(model, batch_size=len())
#     dh_list = np.diff(h_list)
#     dW_list = np.diff()

#     # derivative of function e.g. F = (25) from Tolga
#     G = derivative(kappa(alphas, a_idx), W)
#     A = G @ W.T - W @ G.T
#     I = identity(W.shape[0])
#     # next point from Crank-Nicolson-like scheme
#     W_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * W

#     # same for R and b
#     G = derivative(kappa(alphas, a_idx), R)
#     A = G @ R.T - R @ G.T
#     I = identity(R.shape[0])
#     R_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * R

#     G = derivative(kappa(alphas, a_idx), b)
#     A = G @ b.T - b @ G.T
#     I = identity(b.shape[0])
#     b_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * W

#     return W_next, R_next, b_next
