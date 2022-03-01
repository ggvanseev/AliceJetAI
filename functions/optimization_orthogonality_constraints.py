import torch

from functions.data_manipulation import get_full_pytorch_weight

import numpy as np
import time


def lstm_results(
    lstm_model,
    input_dim,
    train_loader,
    track_jets_train_data,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    pooling="mean",
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
    h_bar_gradient_list = []

    # make sure theta is only generetad once.
    theta = None
    theta_gradients = None

    i = 0
    for x_batch, y_batch in train_loader:
        jet_track_local = track_jets_train_data[i]
        i += 1

        # x_batch as input for lstm, pytorch says shape = [sequence_length, batch_size, n_features]
        # x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(device)
        x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(device)
        y_batch = y_batch.to(device)

        ### Train step
        # set model to train
        # lstm_model.train()  # TODO should this be off so the backward() call in the forward pass does not update the weights?

        # Makes predictions
        hn, theta, theta_gradients = lstm_model(
            x_batch, theta=theta, theta_gradients=theta_gradients
        )

        # get mean/last pooled hidden states
        if pooling == "last":
            h_bar = hn[:, jet_track_local]
        elif pooling == "mean":
            h_bar = torch.zeros([1, len(jet_track_local), hn.shape[-1]])
            jet_track_prev = 0
            for i, jet_track in enumerate(jet_track_local):
                h_bar[:, i] = torch.mean(hn[:, jet_track_prev:jet_track],dim=1)
                jet_track_prev = jet_track
                
                

        # h_bar_list.append(h_bar) # TODO, h_bar is not of fixed length! solution now: append all to list, then vstack the list to get 2 axis structure
        h_bar_list.append(h_bar)

    # Get mean of theta gradients
    for key1, value1 in theta_gradients.items():
        for key2, value2 in value1.items():
            theta_gradients[key1][key2] = theta_gradients[key1][key2] / len(
                train_loader
            )

    return (
        torch.vstack([h_bar[-1] for h_bar in h_bar_list]),
        theta,
        theta_gradients,
    )  # take h_bar[-1], to take the last layer as output


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
        ).cpu()  # Matrix multiplication is time consuming, thus to do this as least as possbile do this
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


def calc_g(gradient_hi, h_bar_list, alphas, a_idx):
    """Calculates the derivative of G to a specific weight or bias.
    G = dkappa / dW_ij = (dkappa * dh_ij) *(dh_ij / dW_ij)
    Since the derivative of x^T x = 2x and
    dh / dW can be obtained from theta_gradients
    this is:

    G = dkappa / dW_ij = (dkappa * dh_ij) *(dh_ij / dW_ij) =
    (0.5*sumi,j alpah_i*alpha_j*2*hi) * dh/dw(theta_gradients)
    = sum_alhpa_j*alpha_i*h_i* dh/dw(theta_gradients)

    proof: https://math.stackexchange.com/questions/1377764/derivative-of-vector-and-vector-transpose-product
    """
    alphas_sum = np.sum(alphas)

    # check device type, and adjust alphas_tensor
    if h_bar_list.device.type == "cpu":
        alphas_tensor = torch.tensor(alphas).type(torch.FloatTensor)
    else:
        alphas_tensor = torch.tensor(alphas).type(torch.FloatTensor).cuda()

    d_kappa = alphas_sum * alphas_tensor @ h_bar_list[a_idx]

    out = (d_kappa * gradient_hi.T).T

    return out


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

            # Deviation from Tolga, normalize to number of h_bar_list TODO
            # g = g / (len(h_bar_list) * 1e-3)

            a = g @ weight.T - weight @ g.T
            i = torch.eye(weight.shape[0], device=device)

            # next point from Crank-Nicolson-like scheme
            track_weights[weight_name] = (
                torch.inverse(i + mu / 2 * a) @ (i - mu / 2 * a) @ weight
            )

        # store in theta
        updated_theta[weight_type] = track_weights

    return updated_theta


def update_lstm(lstm, theta, device):
    """Function to update the weights and bias values of the LSTM

    Args:
        lstm (LSTMModel): the LSTM model
        theta (dict): contains the weights and bias values of the LSTM

    Returns:
        lstm (LSTMModel): the updated LSTM model
    """
    for weight_type, weights in theta.items():
        # get full original weights, called pytorch_weights (following lstm structure)
        pytorch_weights = get_full_pytorch_weight(weights, device)

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
    lstm = update_lstm(lstm, theta, device)

    return lstm, theta
