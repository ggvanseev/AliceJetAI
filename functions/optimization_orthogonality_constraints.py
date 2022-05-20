import torch

from functions.data_manipulation import get_full_pytorch_weight

import numpy as np

from numba import njit


@njit
def kappa_loop_njit(n_alphas, alphas, h_list_adjusted):
    """
    Us njit for speed
    """
    out = 0  # use for trackking summation

    # alphas_matrix = np.outer(alphas, alphas).T  # removed since to much memmory consumption

    for i in range(n_alphas):
        # out += 0.5 * np.dot(alphas_matrix[i], h_matrix[i])
        h_matrix = np.dot(h_list_adjusted, h_list_adjusted[i])
        out += 0.5 * np.dot(alphas[i] * alphas, h_matrix)

    return out


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
        # Due to memmory issues decrease alphas from float64 to float32 , toll
        alphas = alphas.astype(np.float32)

        # h_matrix = (
        #     (h_list[a_idx] @ h_list[a_idx].T).cpu().detach().numpy()
        # )  # Matrix multiplication is time consuming, thus to do this as least as possbile do this
        # once and create a matrix with the results

        # out = 0  # use for trackking summation
        # n_alphas = len(a_idx)

        # alphas_matrix = np.outer(alphas, alphas).T  # removed since to much memmory consumption

        # for i in range(n_alphas):
        #     # out += 0.5 * np.dot(alphas_matrix[i], h_matrix[i])
        #     out += 0.5 * np.dot(alphas[i] * alphas, h_matrix[i])

        out = kappa_loop_njit(
            n_alphas=len(a_idx),
            alphas=alphas,
            h_list_adjusted=h_list[a_idx].cpu().detach().numpy(),
        )

    return out


def time_it():
    time_start = time


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
    # alphas_sum = np.sum(alphas) # TODO this is now always 1 -> normalized!

    # check device type, and adjust alphas_tensor
    if h_bar_list.device.type == "cpu":
        alphas_tensor = torch.tensor(alphas).type(torch.FloatTensor)
    else:
        alphas_tensor = torch.tensor(alphas).type(torch.FloatTensor).cuda()

    d_kappa = alphas_tensor @ h_bar_list[a_idx]  # * alphas_sum

    # seems this one is correct after all...
    out = (d_kappa * gradient_hi.T).T

    # out = 0.5 * (d_kappa * gradient_hi + d_kappa * gradient_hi.T)

    # a = 0
    # for i, idx1 in enumerate(a_idx):
    #     for j, idx2 in enumerate(a_idx):
    #         #a += 0.5 * alphas[i] * alphas[j] * (gradient_hi.T * h_bar_list[idx2] + h_bar_list[idx1].T * gradient_hi)
    #         a += 0.5 * alphas[i] * alphas[j] * (h_bar_list[idx2] + h_bar_list[idx1].T) * gradient_hi

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
            # normalize g to sum of g?
            # g = g / torch.sum(g)
            # remove nan
            # g[g != g] = 0

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
