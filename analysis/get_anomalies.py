import pickle
import os
import numpy as np
import torch
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from functions.data_loader import load_n_filter_data

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"


job_id = "22_04_25_1107"
kt_cut = None

# Load and filter data for criteria eta and jetpt_cap
g_recur_jets, q_recur_jets = load_n_filter_data(file_name, kt_cut=kt_cut)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

trials = trials_test_list["_trials"]

# from run excluded files:
classification_check = CLASSIFICATION_CHECK()
# indices_zero_per_anomaly_nine_flag = classification_check.classification_all_nines_test(
#     trials=trials
# )

# TODO part of testing! to plot in 2D anomalous vs normal data
# make out directory if it does not exist yet
out_dir = f"testing/output/{job_id}"
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

# remove unwanted results:
track_unwanted = list()
for i in range(len(trials)):
    if (
        trials[i]["result"]["loss"] == 10
        or trials[i]["result"]["hyper_parameters"]["num_layers"] == 2
        # or trials[i]["result"]["hyper_parameters"]["scaler_id"] == "minmax"
        # or i in indices_zero_per_anomaly_nine_flag
    ):
        track_unwanted = track_unwanted + [i]

trials = [i for j, i in enumerate(trials) if j not in track_unwanted]

for i in range(2):
    if i == 0:
        jets = q_recur_jets
        type_jets = "quark jet"
    else:
        jets = g_recur_jets
        type_jets = "gluon jet"

    print(f"\nAnomalies for data of type: {type_jets}")
    get_anomalies(jets, job_id, trials, file_name, jet_info=type_jets)

# TODO test of dit iets moois krijgt
# normal_vs_anomaly_2D_all(trials_h_bars, trials_classifications, trials_ocsvms, f"testing/output/{job_id}")
