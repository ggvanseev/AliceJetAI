import pickle
from functions.data_loader import load_n_filter_data, load_anomalies
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    separate_anomalies_from_regular,
)

from plotting.comparison import (
    hist_comparison_first_entries,
    hist_comparison_flatten_entries,
    hist_comparison_non_recur,
    hist_comparison_non_recur_jewel_vs_pythia,
)

import awkward as ak
import numpy as np

import win32gui as wg
from win32gui import GetForegroundWindow
import win32com.client

# Can be multiple jobs, but needs same underlying dataset to be selected upon, i.e. anomalies_info["data"] must be the same for all

job_ids = [
    "10299071",
    "10299072",
    "10299073",
    "10299074",
    "10299075",
    "10299076",
    "10299077",
    "10299078",
    "10299079",
    "10299080",
]


jet_info_pythia = "pythia_50k"
jet_info_jewel = "jewel_100k"
save_flag = True
num = 28
jet_branches = [na.jet_M, na.tau1, na.tau2, na.tau2tau1, na.z2_theta1, na.z2_theta15]

show_distribution_percentages_flag = True


# load data
anomalies_info_pythia = load_anomalies(job_ids=job_ids, jet_info=jet_info_pythia)
anomalies_info_jewel = load_anomalies(job_ids=job_ids, jet_info=jet_info_jewel)

jets_recur_pythia, jets_pythia = load_n_filter_data(
    file_name=anomalies_info_pythia["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
    jet_branches=jet_branches,
)

jets_recur_jewel, jets_jewel = load_n_filter_data(
    file_name=anomalies_info_jewel["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
    jet_branches=jet_branches,
)

# get anomalies and normal out
anomaly_pythia, normal_pythia = separate_anomalies_from_regular(
    anomaly_track=anomalies_info_pythia["classification_annomaly"][num],
    jets_index=anomalies_info_pythia["jets_index"][num],
    data=jets_pythia,
)

anomaly_jewel, normal_jewel = separate_anomalies_from_regular(
    anomaly_track=anomalies_info_jewel["classification_annomaly"][num],
    jets_index=anomalies_info_jewel["jets_index"][num],
    data=jets_jewel,
)

if show_distribution_percentages_flag:
    plt.figure(
        f"Distribution histogram anomalies {jet_info_jewel}", figsize=[1.36 * 8, 8]
    )
    # TODO: note that this is a cut based manual selection before, should maybe be tracked differently in the future..
    filtered_nu_anomalies_info_percentage = (
        anomalies_info_jewel["percentage_anomalies"][
            anomalies_info_jewel["percentage_anomalies"] <= 0.4
        ]
        * 100
    )
    hist_info = plt.hist(filtered_nu_anomalies_info_percentage)

    # Get mean and std
    mean = np.round(np.mean(filtered_nu_anomalies_info_percentage), 2)
    std = np.round(np.std(filtered_nu_anomalies_info_percentage), 2)
    plt.vlines(
        mean,
        color="red",
        ymin=0,
        ymax=np.max(hist_info[0]),
        label=f"Average = {mean} +\- {std} %",
    )
    plt.axvspan(mean - std, mean + std, ymax=0.955, alpha=0.5, color="red")
    # plt.yticks(np.arange(0,hist_info[0]))
    plt.xlabel(f"Percentage (%) jets anomalies {jet_info_jewel}")
    plt.ylabel(f"N")
    plt.legend()
    plt.show()


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
    feature=na.jet_M,
    jet_info=jet_info_jewel,
    n_bins=50,
    save_flag=save_flag,
    job_id="mix_10299081_90",
    num=num,
)
plt.show()
a = 1

for feature in jet_branches:
    hist_comparison_non_recur(
        anomaly=anomaly_pythia,
        normal=normal_pythia,
        feature=feature,
        jet_info=jet_info_pythia,
        n_bins=50,
        save_flag=save_flag,
        job_id="mix_10299081_90",
        num=num,
    )


for feature in jet_branches:
    hist_comparison_non_recur(
        anomaly=anomaly_jewel,
        normal=normal_jewel,
        feature=feature,
        jet_info=jet_info_jewel,
        n_bins=50,
        save_flag=save_flag,
        job_id="mix_10299081_90",
        num=num,
        control={"pythia_normal": normal_pythia, "pythia_anomaly": anomaly_pythia},
    )

for feature in jet_branches:
    hist_comparison_non_recur_jewel_vs_pythia(
        jewel=normal_jewel,
        pythia=normal_pythia,
        feature=feature,
        jet_info="normal_v_normal",
        n_bins=50,
        save_flag=save_flag,
        job_id="mix_10299081_90",
        num=num,
    )

for feature in jet_branches:
    hist_comparison_non_recur_jewel_vs_pythia(
        jewel=anomaly_jewel,
        pythia=anomaly_pythia,
        feature=feature,
        jet_info="anomaly_v_anomaly",
        n_bins=50,
        save_flag=save_flag,
        job_id="mix_10299081_90",
        num=num,
    )


# manual filter based on visual jet mass
aw = np.zeros(1000 + 1)
selected_list = list()
aw[0] = GetForegroundWindow()
for num in list(anomalies_info_pythia["classification_annomaly"].keys()):
    # give user the option to select using the terminal
    # get anomalies and normal out
    anomaly_pythia, normal_pythia = separate_anomalies_from_regular(
        anomaly_track=anomalies_info_pythia["classification_annomaly"][num],
        jets_index=anomalies_info_pythia["jets_index"][num],
        data=jets_pythia,
    )

    anomaly_jewel, normal_jewel = separate_anomalies_from_regular(
        anomaly_track=anomalies_info_jewel["classification_annomaly"][num],
        jets_index=anomalies_info_jewel["jets_index"][num],
        data=jets_jewel,
    )

    # This gets the details of the current window, the one running the program
    shell = win32com.client.Dispatch("WScript.Shell")

    hist_comparison_non_recur_jewel_vs_pythia(
        jewel=normal_jewel,
        pythia=normal_pythia,
        feature=na.jet_M,
        jet_info="normal_v_normal",
        n_bins=50,
        save_flag=False,
        job_id="mix_10299081_90",
        num=num,
    )
    continue_flag = True
    while continue_flag:
        shell.SendKeys("%")
        wg.SetForegroundWindow(aw[num])
        get_pass = input("y or n:")
        if get_pass:
            aw[num + 1] = GetForegroundWindow()
        if get_pass == "y":
            selected_list.append(num)
            continue_flag = False
        elif get_pass == "n":
            continue_flag = False
    plt.close()


    