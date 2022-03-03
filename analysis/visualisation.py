import pickle
from functions.data_loader import load_n_filter_data
import branch_names as na
import matplotlib.pyplot as plt
from functions.data_manipulation import (
    seperate_anomalies_from_regular,
    format_ak_to_list,
)

job_id = 9727358
num = 0

# load data
anomalies_info = pickle.load(
    open(f"storing_results/anomaly_classification_{job_id}.pkl", "rb")
)

_, _, g_recur_jets, q_recur_jets = load_n_filter_data(
    file_name=anomalies_info["file"],
    jet_recur_branches=[na.recur_dr, na.recur_jetpt, na.recur_z],
)

# reformat to list
q_recur_jets = format_ak_to_list(q_recur_jets)

# get anomalies and normal out
q_anomaly, q_normal = seperate_anomalies_from_regular(
    anomaly_track=anomalies_info["classifaction_annomaly"][num],
    jets_index=anomalies_info["jets_index"][num],
    data=q_recur_jets,
)

a = 1
