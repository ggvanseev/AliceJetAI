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


job_id = "14_03_22_1028"


# Load and filter data for criteria eta and jetpt_cap
_, _, _, q_recur_jets = load_n_filter_data(file_name, kt_cut=False)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

trials = trials_test_list["_trials"]

# from run excluded files:
classifaction_check = CLASSIFICATION_CHECK()
indices_zero_per_anomaly_nine_flag = classifaction_check.classifaction_all_nines_test(
    trials=trials
)

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

anomaly_tracker = np.zeros(len(trials))
for i in range(len(trials)):
    # select model
    model = trials[i]["result"]["model"]

    lstm_model = model["lstm"]  # note in some old files it is lstm:
    ocsvm_model = model["ocsvm"]
    scaler = model["scaler"]

    # get hyper parameters
    batch_size = int(trials[i]["result"]["hyper_parameters"]["batch_size"])
    input_variables = list(trials[i]["result"]["hyper_parameters"]["variables"])

    classifier = LSTM_OCSVM_CLASSIFIER(
        oc_svm=ocsvm_model, lstm=lstm_model, batch_size=batch_size, scaler=scaler
    )

    classifaction, anomaly_tracker[i], _ = classifier.anomaly_classifaction(
        data=q_recur_jets[input_variables]
    )

    # print(
    #     f"Percentage classified as anomaly: {np.round(anomaly_tracker[i]*100,2) }%, where the model has a nu of {trials[i]['result']['hyper_parameters']['svm_nu']}"
    # )
    print(
        f"Anomalous: {np.round(anomaly_tracker[i]*100,2)}%,\tnu: {trials[i]['result']['hyper_parameters']['svm_nu']},\tLoss: {trials[i]['result']['loss']:.2E},  \tCost: {trials[i]['result']['final_cost']:.2E}"
    )
print(
    f"Average percentage anomalies: {np.round(np.nanmean(anomaly_tracker)*100,2)} +\- {np.round(np.nanstd(anomaly_tracker)*100,2)}%"
)

a = 1
