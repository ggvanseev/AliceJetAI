from scipy.sparse import identity  # use sparse matrices for memory efficiency TODO

from functions.data_manipulation import get_weights


def derivative():
    return


def kappa(alphas, a_idx, h_list):
    out = 0
    for idx1, i in enumerate(a_idx):
        for idx2, j in enumerate(a_idx):
            out += 0.5 * alphas[i] * alphas[j] * (h_list[i].T @ h_list[j])
    return out


def optimization(model, h_list, alphas, a_idx, mu):

    # obtain W, R and b from current h
    # W = h.parameters
    # R = h.parameters
    # b = h.parameters
    W, R, b = get_weights(model, batch_size=len())
    dh_list = np.diff(h_list)
    dW_list = np.diff()

    # derivative of function e.g. F = (25) from Tolga
    G = derivative(kappa(alphas, a_idx), W)
    A = G @ W.T - W @ G.T
    I = identity(W.shape[0])
    # next point from Crank-Nicolson-like scheme
    W_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * W

    # same for R and b
    G = derivative(kappa(alphas, a_idx), R)
    A = G @ R.T - R @ G.T
    I = identity(R.shape[0])
    R_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * R

    G = derivative(kappa(alphas, a_idx), b)
    A = G @ b.T - b @ G.T
    I = identity(b.shape[0])
    b_next = (I + mu / 2 * A) ** (-1) * (I - mu / 2) * W

    return W_next, R_next, b_next
