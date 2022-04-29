import pickle
from functions.data_loader import load_n_filter_data
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    separate_anomalies_from_regular,
)

import awkward as ak
import numpy as np

job_id = 9792527
jet_info = "quark"
num = 0

# load data
anomalies_info = pickle.load(
    open(f"storing_results_1/anomaly_classification_{jet_info}_{job_id}.pkl", "rb")
)


g_recur_jets, q_recur_jets = load_n_filter_data(
    file_name=anomalies_info["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
)

# get anomalies and normal out
q_anomaly, q_normal = separate_anomalies_from_regular(
    anomaly_track=anomalies_info["classification_annomaly"][num],
    jets_index=anomalies_info["jets_index"][num],
    data=q_recur_jets,
)

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


def hist_comparison(
    anomaly: np,
    normal: np,
    feature: str,
    jet_info=None,
    n_bins=50,
):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(1.36 * 8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Ensure same bin-size.
    dist_combined, bins = np.histogram(
        np.hstack((normal, anomaly)), bins=n_bins, density=True
    )

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.99, wspace=0.4, hspace=0)

    # plot top picture, showing
    ax[0].hist(
        normal,
        bins,
        label="normal",
        density=True,
        color="blue",
        histtype="step",
    )

    ax[0].hist(
        anomaly,
        bins,
        label="anomaly",
        density=True,
        color="red",
        histtype="step",
    )

    ax[0].hist(
        np.hstack((normal, anomaly)),
        bins,
        label="combined",
        density=True,
        color="black",
        histtype="step",
    )
    ax[0].set_ylabel("N")

    # plot ratio
    dist_normal = np.histogram(normal, bins, density=True)[0]
    dist_anomaly = np.histogram(anomaly, bins, density=True)[0]

    ratio_combined = dist_combined / dist_combined
    ratio_normal = dist_normal / dist_combined
    ratio_anomaly = dist_anomaly / dist_combined

    ax[1].plot(bins[1:], ratio_combined, color="black")
    ax[1].plot(bins[1:], ratio_anomaly, color="red")
    ax[1].plot(bins[1:], ratio_normal, color="blue")

    ax[1].set_xlabel(feature)
    ax[1].set_ylabel("ratio")

    max = (
        np.nanmax(ratio_anomaly[1:])
        if np.nanmax(ratio_anomaly[1:]) > np.nanmax(ratio_normal[1:])
        else np.nanmax(ratio_normal[1:])
    )

    ax[1].set_ylim([0, max])
    ax[1].set_xlim([0, bins[-1] + bins[1]])
    ax[0].legend()

    # move spines
    ax[0].spines["left"].set_position(("data", 0.0))
    ax[1].spines["left"].set_position(("data", 0.0))


def hist_comparison_first_entries(
    anomaly: ak, normal: ak, feature: str, jet_info=None, n_bins=50
):
    hist_comparison(
        anomaly=ak.to_numpy(ak.firsts(anomaly[feature])),
        normal=ak.to_numpy(ak.firsts(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
    )


def hist_comparison_flatten_entries(
    anomaly: ak, normal: ak, feature: str, jet_info=None, n_bins=50
):
    hist_comparison(
        anomaly=ak.to_numpy(ak.flatten(anomaly[feature])),
        normal=ak.to_numpy(ak.flatten(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
    )


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
    plt.ylabel("N")
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


stacked_plot_normalised_first_entries(
    anomaly=q_anomaly, normal=q_normal, feature=na.recur_dr, jet_info=None, n_bins=50
)
plt.show()
a = 1
