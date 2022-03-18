import pickle
import pandas as pd
import sklearn.svm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from functions.classification import LSTM_OCSVM_CLASSIFIER, CLASSIFICATION_CHECK
from functions.data_manipulation import format_ak_to_list
from functions.data_loader import load_n_filter_data
from plotting.general import plot_cost_vs_cost_condition

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"


job_id = 9737619


# Load and filter data for criteria eta and jetpt_cap
g_recur_jets, q_recur_jets = load_n_filter_data(file_name)

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# trials_test_list = torch.load(
#    f"storing_results/trials_test_{job_id}.p", map_location=device
# )
trials = pickle.load(open(r"storing_results\trials_train_9737619.p", "rb"))


anomaly_tracker = np.zeros(len(trials))
for i in range(len(trials)):
    # select model
    model = trials[i]["model"]

    lstm_model = model["lstm:"]  # note in some old files it is lstm:
    ocsvm_model = model["ocsvm"]
    scaler = model["scaler"]

    # get hyper parameters
    batch_size = int(trials[i]["hyper_parameters"]["batch_size"])
    # input_variables = list(trials[i]["hyper_parameters"]["variables"])

    classifier = LSTM_OCSVM_CLASSIFIER(
        oc_svm=ocsvm_model, lstm=lstm_model, batch_size=batch_size, scaler=scaler
    )

    classifaction, anomaly_tracker[i], _ = classifier.anomaly_classifaction(
        data=g_recur_jets
    )

    print(f"Percentage classified as anomaly: {np.round(anomaly_tracker[i]*100,2) }%")

print(
    f"Average percentage anomalys: {np.round(np.nanmean(anomaly_tracker)*100,2)} +\- {np.round(np.nanstd(anomaly_tracker)*100,2)}%"
)

plot_cost_vs_cost_condition(
    track_cost=trials[0]["cost_data"]["cost"],
    track_cost_condition=trials[0]["cost_data"]["cost_condition"],
    title_plot="",
    show_flag=True,
)

a = 1
