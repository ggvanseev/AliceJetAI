import pickle
import pandas as pd
import sklearn.svm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from ai.lstm_ocsvm_class import LSTM_OCSVM_CLASSIFIER, CLASSIFICATION_CHECK
from functions.data_manipulation import format_ak_to_list
from functions.data_loader import load_n_filter_data

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"


job_id = 9727358


# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, q_recur_jets = load_n_filter_data(file_name)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

trials = trials_test_list["_trials"]

# from run excluded files:
classifaction_check = CLASSIFICATION_CHECK()
# indices_zero_per_anomaly_nine_flag = classifaction_check.classifaction_all_nines_test(
#     trials=trials
# )

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
        type_jets = "quark"
    else:
        jets = g_recur_jets
        type_jets = "gluon"
    anomaly_tracker = np.zeros(len(trials))
    classifaction_tracker = dict()
    jets_index_tracker = dict()
    jets_tracker = dict()
    for i in range(len(trials)):
        # select model
        model = trials[i]["result"]["model"]

        lstm_model = model["lstm:"]  # note in some old files it is lstm:
        ocsvm_model = model["ocsvm"]
        scaler = model["scaler"]

        # get hyper parameters
        batch_size = int(trials[i]["result"]["hyper_parameters"]["batch_size"])
        # input_variables = list(trials[i]["result"]["hyper_parameters"]["variables"])

        classifier = LSTM_OCSVM_CLASSIFIER(
            oc_svm=ocsvm_model, lstm=lstm_model, batch_size=batch_size, scaler=scaler
        )

        (
            classifaction_tracker[i],
            anomaly_tracker[i],
            jets_index_tracker[i],
            jets_tracker[i],
        ) = classifier.anomaly_classifaction(
            data=jets  # [input_variables]
        )

        print(
            f"Percentage classified as anomaly: {np.round(anomaly_tracker[i]*100,2) }%, where the model has a nu of {trials[i]['result']['hyper_parameters']['svm_nu']}"
        )

    print(
        f"Average percentage anomalys: {np.round(np.nanmean(anomaly_tracker)*100,2)} +\- {np.round(np.nanstd(anomaly_tracker)*100,2)}%"
    )

    # store results
    storing = {
        "jets_index": jets_index_tracker,
        "percentage_anomalies": anomaly_tracker,
        "classifaction_annomaly": classifaction_tracker,
        "jets": jets_tracker,
        "file": file_name,
    }
    pickle.dump(
        storing,
        open(f"storing_results/anomaly_classification_{type_jets}_{job_id}.pkl", "wb"),
    )

a = 1
