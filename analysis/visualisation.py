import pickle
from functions.data_loader import load_n_filter_data
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    seperate_anomalies_from_regular,
    format_ak_to_list,
)

job_id = 9792527
jet_info = "gluon"
num = 0

# load data
anomalies_info = pickle.load(
    open(f"storing_results/anomaly_classification_{jet_info}_{job_id}.pkl", "rb")
)


g_recur_jets, q_recur_jets = load_n_filter_data(
    file_name=anomalies_info["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
)

# get anomalies and normal out
q_anomaly, q_normal = seperate_anomalies_from_regular(
    anomaly_track=anomalies_info["classifaction_annomaly"][num],
    jets_index=anomalies_info["jets_index"][num],
    data=q_recur_jets,
)

plt.figure(f"Distribution histogram anomalies {jet_info}", figsize=[1.36 * 8, 8])
plt.hist(anomalies_info["percentage_anomalies"])
plt.xlabel(f"Percentage (%) jets anomalies {jet_info}")
plt.ylabel(f"N")

a = 1
