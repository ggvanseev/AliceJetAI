import pickle
from functions.data_loader import load_n_filter_data
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    separate_anomalies_from_regular,
)

from plotting.comparison import (
    hist_comparison_first_entries,
    hist_comparison_flatten_entries,
)

import awkward as ak
import numpy as np

job_id = 10206558
jet_info = "pythia_5k"
save_flag = True
num = 21

show_distribution_percentages_flag = False


# load data
anomalies_info = pickle.load(
    open(f"storing_results/anomaly_classification_{jet_info}_{job_id}.pkl", "rb")
)


recur_jets, jets = load_n_filter_data(
    file_name=anomalies_info["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
)

# get anomalies and normal out
anomaly, normal = separate_anomalies_from_regular(
    anomaly_track=anomalies_info["classification_annomaly"][num],
    jets_index=anomalies_info["jets_index"][num],
    data=recur_jets,
)

if show_distribution_percentages_flag:
    plt.figure(f"Distribution histogram anomalies {jet_info}", figsize=[1.36 * 8, 8])
    plt.hist(anomalies_info["percentage_anomalies"])
    plt.xlabel(f"Percentage (%) jets anomalies {jet_info}")
    plt.ylabel(f"N")


def set_axis_at_origin(ax):
    # set the x-spine
    ax.spines["left"].set_position("zero")

    # turn off the right spine/ticks
    ax.spines["right"].set_color("none")
    ax.yaxis.tick_left()
    ax.set_ylabel("y", fontsize=16)
    ax.yaxis.set_label_coords(0.49, 1)

    # set the y-spine
    ax.spines["bottom"].set_position("zero")

    # turn off the top spine/ticks
    ax.spines["top"].set_color("none")
    ax.xaxis.tick_bottom()
    ax.set_xlabel("x", fontsize=16)
    ax.xaxis.set_label_coords(1, 0.5)


hist_comparison_first_entries(
    anomaly=anomaly,
    normal=normal,
    feature=na.recur_dr,
    jet_info=jet_info,
    n_bins=50,
    save_flag=save_flag,
    job_id=job_id,
    num=num,
)
plt.show()
a = 1
