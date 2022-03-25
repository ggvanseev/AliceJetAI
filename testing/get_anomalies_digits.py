import pickle
import numpy as np
import torch
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from testing_functions import load_digits_data

from  testing.plotting import normal_vs_anomaly_2D_all


job_id = "22_03_24_1808"

# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"

file_name = test_file
print_dataset_info = False


# get digits data
train_dict = load_digits_data(train_file, print_dataset_info=print_dataset_info)
test_dict = load_digits_data(test_file)

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

trials = trials_test_list["_trials"] # TODO new!
# trials = {i: {"result": j} for i, j in trials_test_list.items()}  # TODO for old results

# from run excluded files:
classification_check = CLASSIFICATION_CHECK()
# indices_zero_per_anomaly_nine_flag = classification_check.classification_all_nines_test(
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

# trials = [i for j, i in trials.items() if j not in track_unwanted] # TODO old version
trials = [i for j, i in enumerate(trials) if j not in track_unwanted]

trials_h_bars = dict()
trials_classifications = dict()

for i in range(3):
    if i == 0:
        jets = test_dict["0"][:200]
        type_jets = "0_digits"
    elif i == 1:
        jets = test_dict["9"][:200]
        type_jets = "9_digits"
    else:
        jets = test_dict["8"][:200]
        type_jets = "8_digits"

    print(f"Anomalies for data of type: {type_jets}")
    trials_h_bars[type_jets], trials_classifications[type_jets] = get_anomalies(jets, job_id, trials, file_name, jet_info=type_jets)
    
    print("\n")
    

normal_vs_anomaly_2D_all(trials_h_bars, trials_classifications, f"testing/output/{job_id}")
