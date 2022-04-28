"""
File which stores functions related to plotting the svm boundary in 2D.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import awkward as ak
import torch

from functions.data_manipulation import branch_filler, lstm_data_prep, h_bar_list_to_numpy, format_ak_to_list
from functions.run_lstm import calc_lstm_results


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
    

def svm_boundary_train_plot(model, X1, out_file, title=r"$\overline{h_i}$", y=None, fit=False, ax=plt):
    
    if fit:
        # fit (train) and predict the data, if y
        model.fit(X1, y, sample_weight=None) # TODO figure out sample_weight
        
    pred = model.predict(X1)
    
    xmin = X1[:,0].min()
    xmax = X1[:,0].max()
    ymin = X1[:,1].min()
    ymax = X1[:,1].max()
    
    margin_x = 0.05 * (xmax-xmin)
    margin_y = 0.05 * (ymax-ymin)
    
    # meshgrid for plots
    xx1, yy1 = np.meshgrid(np.linspace(xmin - margin_x, xmax + margin_x, 500),
                            np.linspace(ymin - margin_y, ymax + margin_y, 500))
    
    # decision function
    Z1 = model.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    
    # plot data and decision function
    ax.scatter(X1[:, 0], X1[:, 1], c=pred, cmap=plt.cm.viridis, alpha=0.5, label="Training Data")
    ax.contour(xx1, yy1, Z1, levels=(-1,0,1), linewidths=(0.5, 0.75, 0.5),
                linestyles=('--', '-', '--'), colors=['k','k','k'])
    ax.contourf(xx1, yy1, Z1, cmap=cm.get_cmap("coolwarm_r"), alpha=0.5, linestyles="None")
    

    # Plot support vectors (non-zero alphas)
    # as circled points (linewidth > 0)
    # TODO y should be the labels -> quark or gluon decided by pythia, 0 or 9 decided by digits
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c=y[model.support_] if y else model.predict(model.support_vectors_),
                cmap=plt.cm.viridis, lw=1, edgecolors='k', label="Support Vectors")
    
    plt.title(title)
    plt.xlabel(r"$\overline{h_{i,1}}$")
    plt.ylabel(r"$\overline{h_{i,2}}$")
    
    ax.legend(prop={'size': 8})
    ax.margins(1.5,1.5)

    # save and close the plot
    ax.savefig(out_file)
    plt.close()  # close figure - clean memory


def svm_boundary_plots(trials: dict, job_id, train_data, labels=None, fit=False):
    # make out directory if it does not exist yet
    out_dir = f"output/svm_boundary_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # store trial parameters
    out_txt = ""
    
    if type(train_data) == ak.highlevel.Array:
        n_features = len(train_data.fields)
    else:
        n_features = len(train_data[0][0])

    # build plots for each trial
    for i, trial in enumerate(trials["_trials"]):
        out_txt += f"trial: {i}"

        # obtain results from the trial
        result = trial["result"]
        model = result["model"]
        lstm = model["lstm"]
        ocsvm = model["ocsvm"]
        scaler = model["scaler"]

        # extract hyper parameters from the results
        h_parm = result["hyper_parameters"]
        title_plot = f""
        for key in h_parm:
            title_plot += f"{h_parm[key]}_{key}_"
            out_txt += "\n  {:12}\t  {}".format(key, h_parm[key])

        # add final loss and cost to the info
        out_txt += f"\n{'with loss:':18}{result['loss']}"
        out_txt += f"\n{'with final cost:':18}{result['final_cost']}"
        
        # check if ak or if list
        if type(train_data) is not list:
            train_data = format_ak_to_list(train_data)
        
        # put train_data in batches again
        train_batches, track_jets_in_batch, _, _ = branch_filler(train_data, h_parm["batch_size"], n_features=n_features)
        
        # obtain scaler from result and put train data in train_loader
        scaler = result['model']['scaler']
        train_loader = lstm_data_prep(
            data=train_batches, scaler=scaler, batch_size=h_parm["batch_size"], fit_flag=False
        )
        
        # get h_bar states
        h_bar_list, _, _ = calc_lstm_results(lstm, len(train_batches[0]), train_loader, track_jets_in_batch)
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, torch.device("cpu"))
        
        # generate the plot
        fig = svm_boundary_train_plot(ocsvm, h_bar_list_np, out_file=out_dir+ "/" f"trial_{i}.png", title=r"$\overline{h_i}$" + f" train states - trial_{i}", y=labels,fit=fit)
        if fig == -1:
            break
