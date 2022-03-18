import pickle
import pandas as pd
import sklearn.svm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from ai.lstm_ocsvm_class import LSTM_OCSVM_CLASSIFIER, CLASSIFICATION_CHECK
from functions.data_manipulation import format_ak_to_list
from functions.data_loader import load_n_filter_data, load_digits_data


job_id = "18_03_22_1428"

# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"

file_name = test_file

print_dataset_info = True
# get digits data
train_dict = load_digits_data(train_file, print_dataset_info=print_dataset_info)
test_dict = load_digits_data(test_file)

# plot random sample
# plt.figure()
# plt.scatter(*train_dict["8"][24].T)
# plt.xlim(0,500)
# plt.ylim(0,500)
# plt.show()

# mix "0" = 90% as normal data with "9" = 10% as anomalous data
# train_data = train_dict["0"][:675] + train_dict["9"][:75]
# print('Mixed "0": 675 = 90% of normal data with "9": 75 = 10% as anomalous data for a train set of 750 samples')
# test_data = test_dict["0"][:360] + test_dict["9"][:40]
# print('Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples')

# q_recur_jets = (np.zeros([500, 10, 3])).tolist()

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

trials = trials_test_list

# from run excluded files:
classifaction_check = CLASSIFICATION_CHECK()
# indices_zero_per_anomaly_nine_flag = classifaction_check.classifaction_all_nines_test(
#     trials=trials
# )

# remove unwanted results:
track_unwanted = list()
for i in range(len(trials)):
    if (
        trials[i]["loss"] == 10
        or trials[i]["hyper_parameters"]["num_layers"] == 2
        # or trials[i]["hyper_parameters"]["scaler_id"] == "minmax"
        # or i in indices_zero_per_anomaly_nine_flag
    ):
        track_unwanted = track_unwanted + [i]

trials = [i for j, i in trials.items() if j not in track_unwanted]

for i in range(2):
    if i == 0:
        jets = test_dict["0"][:100]
        type_jets = "0"
    else:
        jets = test_dict["9"][:100]
        type_jets = "9"
    print(f"\nFor jets of type: {type_jets}")
    anomaly_tracker = np.zeros(len(trials))
    classifaction_tracker = dict()
    jets_index_tracker = dict()
    jets_tracker = dict()
    for i in range(len(trials)):
        # select model
        model = trials[i]["model"]

        lstm_model = model["lstm"]  # note in some old files it is lstm:
        ocsvm_model = model["ocsvm"]
        scaler = model["scaler"]

        # get hyper parameters
        batch_size = int(trials[i]["hyper_parameters"]["batch_size"])
        # input_variables = list(trials[i]["hyper_parameters"]["variables"])

        classifier = LSTM_OCSVM_CLASSIFIER(
            oc_svm=ocsvm_model, lstm=lstm_model, batch_size=batch_size, scaler=scaler
        )

        (
            classifaction_tracker[i],
            anomaly_tracker[i],
            jets_index_tracker[i],
        ) = classifier.anomaly_classifaction(
            data=jets  # [input_variables]
        )

        print(
            f"Percentage classified as anomaly: {np.round(anomaly_tracker[i]*100,2) }%, where the model has a nu of {trials[i]['hyper_parameters']['svm_nu']}"
        )

    print(
        f"Average percentage anomalies: {np.round(np.nanmean(anomaly_tracker)*100,2)} +\- {np.round(np.nanstd(anomaly_tracker)*100,2)}%"
    )

    # store results
    storing = {
        "jets_index": jets_index_tracker,
        "percentage_anomalies": anomaly_tracker,
        "classifaction_annomaly": classifaction_tracker,
        "file": file_name,
    }
    pickle.dump(
        storing,
        open(f"storing_results/anomaly_classification_{type_jets}_{job_id}.pkl", "wb"),
    )

a = 1
