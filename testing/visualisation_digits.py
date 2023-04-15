"""
Stacked plots to visualize more things on digits dataset.
"""

import pickle
from functions.data_loader import load_n_filter_data
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    separate_anomalies_from_regular,
)

from testing_functions import load_digits_data

import awkward as ak
import numpy as np

job_id = "21_03_22_1650"
jet_info = "9 digits"
num = 0

# load data
anomalies_info = pickle.load(
    open(f"storing_results/anomaly_classification_{jet_info}_{job_id}.pkl", "rb")
)

# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"
file_name = train_file + "," + test_file

# get digits data
train_dict = load_digits_data(train_file)
test_dict = load_digits_data(test_file)

train_data = train_dict["9"][:75] + train_dict["0"][:675]
print(
    'Mixed "0": 675 = 90% of normal data with "9": 75 = 10% as anomalous data for a train set of 750 samples'
)
test_data = test_dict["9"][:40] + test_dict["0"][:360]
print(
    'Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples'
)

# get anomalies and normal out
q_anomaly, q_normal = separate_anomalies_from_regular(
    anomaly_track=anomalies_info["classification_annomaly"][num],
    jets_index=anomalies_info["jets_index"][num],
    data=test_data,
)

plt.figure(f"Distribution histogram anomalies {jet_info}", figsize=[1.36 * 8, 8])
plt.hist(anomalies_info["percentage_anomalies"])
plt.xlabel(f"Percentage (%) jets anomalies {jet_info}")
plt.ylabel(f"N")


def stacked_plot_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
    )
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()


def stacked_plot_normalised_to_max_first_entries(
    anomaly, normal, feature, jet_info=None
):
    norm_normal = ak.firsts(normal[feature]) / np.max(ak.firsts(normal[feature]))
    norm_anomaly = ak.firsts(anomaly[feature]) / np.max(ak.firsts(anomaly[feature]))

    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [norm_normal, norm_anomaly],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [
            ak.firsts(normal[feature])[: len(ak.firsts(anomaly[feature]))],
            ak.firsts(anomaly[feature]),
        ],
        stacked=True,
        label=[f"normal", "anomalous"],
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_normalised_first_entries(
    anomaly, normal, feature, jet_info=None
):
    plt.figure(
        f"Distribution histogram normalised anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [
            ak.firsts(normal[feature])[: len(ak.firsts(anomaly[feature]))],
            ak.firsts(anomaly[feature]),
        ],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_normalised_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


a = 1
