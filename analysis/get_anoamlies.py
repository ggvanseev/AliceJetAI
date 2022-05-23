import pickle
import numpy as np
import torch
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from functions.data_loader import load_n_filter_data

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/pythia_50k_testing.root"


job_ids = [
    "10299081",
    "10299082",
    "10299083",
    "10299084",
    "10299085",
    "10299086",
    "10299087",
    "10299088",
    "10299089",
    "10299090",
]
jet_info = "pythia_50k"
kt_cut = None

print(jet_info, "\n", job_ids)

# Load and filter data for criteria eta and jetpt_cap
recur_jets, jets = load_n_filter_data(file_name, kt_cut=kt_cut)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trials_test_list = [
    torch.load(
        f"storing_results/manual_selected/trials_test_manual_filter_{job_id}.p",
        map_location=device,
    )
    for job_id in job_ids
]
for job_id, trials in zip(job_ids, trials_test_list):

    trials = trials["_trials"]

    # from run excluded files:
    classifaction_check = CLASSIFICATION_CHECK()
    indices_zero_per_anomaly_nine_flag = (
        classifaction_check.classification_all_nines_test(trials=trials)
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

    get_anomalies(recur_jets, job_id, trials, file_name, jet_info)
