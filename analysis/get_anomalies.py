import pickle
import numpy as np
import torch
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg

# file_name(s) - comment/uncomment when switching between local/Nikhef
#file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"
#file_name = "samples/time_cluster_5k.root"


job_id = "10214090"
type_jets = "q_jets"

# Load and filter data for criteria eta and jetpt_cap
g_recur_jets, q_recur_jets, _, _ = load_n_filter_data_qg(file_name, kt_cut=False)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

trials = trials_test_list["_trials"]

# from run excluded files:
classifaction_check = CLASSIFICATION_CHECK()
indices_zero_per_anomaly_nine_flag = classifaction_check.classification_all_nines_test(
    trials=trials
)

# remove unwanted results:
track_unwanted = list()
for i in range(len(trials)):
    if (
        trials[i]["result"]["loss"] == 10
        or trials[i]["result"]["hyper_parameters"]["num_layers"] == 2
        # or trials[i]["result"]["hyper_parameters"]["scaler_id"] == "minmax"
        or i in indices_zero_per_anomaly_nine_flag
    ):
        track_unwanted = track_unwanted + [i]

trials = [i for j, i in enumerate(trials) if j not in track_unwanted]

get_anomalies(q_recur_jets, job_id, trials, file_name, jet_info=type_jets)
