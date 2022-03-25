"""
File which stores functions related to plotting the svm boundary in 2D.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_svm_boundary_2d(h_bar: np, h_predicted: np, svm_model):
    """
    Plots the svm boundary and for a given h_bar from the lstm and an h_predicted by the svm.
    Note: 2d thus the hidden dim for the lstm must be 2
    """

    # define the meshgrid
    x_min, x_max = h_bar[:, 0].min() - 1, h_bar[:, 0].max() + 1
    y_min, y_max = h_bar[:, 1].min() - 1, h_bar[:, 1].max() + 1

    x_ = np.linspace(x_min, x_max, 500)
    y_ = np.linspace(y_min, y_max, 500)

    xx, yy = np.meshgrid(x_, y_)

    # evaluate the decision function on the meshgrid
    # z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = svm_model.decision_function(np.c_[[xx.ravel()] * h_bar.shape[1]].T)
    z = z.reshape(xx.shape)

    # plot the decision function and the reduced data
    plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors="darkred")
    b = plt.scatter(
        h_bar[h_predicted == 1, 0],
        h_bar[h_predicted == 1, 1],
        c="white",
        edgecolors="k",
    )
    c = plt.scatter(
        h_bar[h_predicted == -1, 0],
        h_bar[h_predicted == -1, 1],
        c="gold",
        edgecolors="k",
    )
    plt.legend(
        [a.collections[0], b, c],
        ["learned frontier", "regular observations", "abnormal observations"],
        bbox_to_anchor=(1.05, 1),
    )
    plt.axis("tight")
    plt.show()
