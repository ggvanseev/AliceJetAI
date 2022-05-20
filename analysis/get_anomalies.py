import pickle
import numpy as np
import torch
import branch_names as na
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg
from plotting.stacked import *
from functions.data_manipulation import (
    separate_anomalies_from_regular,
)

# file_name(s) - comment/uncomment when switching between local/Nikhef
#file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"
#file_name = "samples/time_cluster_5k.root"
#file_name = "samples/mixed_1500jets_pct:90g_10q.p"


job_id = "22_05_19_1534"
g_percentage = 90
num = 0
save_flag = True
show_distribution_percentages_flag = False


def tpr_fpr_zero_division(pos, neg):
    """Check if positive is 0, otherwise return ratio"""
    if (pos == 0):
        return 0
    return pos / (pos + neg)
    

# roc curve function
def ROC_curve_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    """Make ROC curve for given features for quarks and gluons"""
    
    # store stacked plots in designated directory
    out_dir = f"output/ROC_curves"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    for feature in features:
        minimum = ak.min((g_anomaly[feature], g_normal[feature], q_anomaly[feature], q_normal[feature]))
        maximum = ak.max((g_anomaly[feature], g_normal[feature], q_anomaly[feature], q_normal[feature]))
        
        tpr = list() # True Positive Rate
        fpr = list() # False Positive Rate
        for i in np.linspace(minimum, maximum, 100):
            g_anomaly_under_th = ak.count_nonzero(g_anomaly[feature][g_anomaly[feature] <= i]) # false negative
            g_normal_under_th = ak.count_nonzero(g_normal[feature][g_normal[feature] <= i]) # true positive
            
            q_anomaly_under_th = ak.count_nonzero(q_anomaly[feature][q_anomaly[feature] <= i]) # true negative
            q_normal_under_th = ak.count_nonzero(q_normal[feature][q_normal[feature] <= i]) # false positive
            
            tpr.append(tpr_fpr_zero_division(g_normal_under_th, g_anomaly_under_th))
            fpr.append(tpr_fpr_zero_division(q_normal_under_th, q_anomaly_under_th))
            
            plt.title(f"ROC curve Gluon vs Quark - {job_id} - {feature}")
            plt.scatter(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            
        
    
    
    return

# Load and filter data for criteria eta and jetpt_cap
g_recur_jets, q_recur_jets, _, _ = load_n_filter_data_qg(file_name, kt_cut=False)
# q_recur_jets = (np.zeros([500, 10, 3])).tolist()


# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# New stuff
# g_jets_recur = torch.load(file_name, map_location=device)

# # split data into quark and gluon jets
# total = len(g_jets_recur)
# q_jets_recur = g_jets_recur[int(total / 100 * g_percentage):]
# g_jets_recur = g_jets_recur[:int(total / 100 * (100-g_percentage))]


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

# gluon jets, get anomalies and normal out
type_jets = "g_jets"
_, g_jets_index_tracker, g_classification_tracker = get_anomalies(g_recur_jets, job_id, trials, file_name, jet_info=type_jets)
g_anomaly, g_normal = separate_anomalies_from_regular(
    anomaly_track=g_classification_tracker[num],
    jets_index=g_jets_index_tracker[num],
    data=g_recur_jets,
)

# quark jets, get anomalies and normal out
type_jets = "q_jets"
_, q_jets_index_tracker, q_classification_tracker = get_anomalies(q_recur_jets, job_id, trials, file_name, jet_info=type_jets)
q_anomaly, q_normal = separate_anomalies_from_regular(
    anomaly_track=q_classification_tracker[num],
    jets_index=q_jets_index_tracker[num],
    data=q_recur_jets,
)

if show_distribution_percentages_flag:
    plt.figure(f"Distribution histogram anomalies {jet_info}", figsize=[1.36 * 8, 8])
    plt.hist(anomalies_info["percentage_anomalies"])
    plt.xlabel(f"Percentage (%) jets anomalies {jet_info}")
    plt.ylabel(f"N")

features = [na.recur_jetpt, na.recur_dr, na.recur_z]
ROC_curve_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_last_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_normalised_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_all_splits_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_last_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_normalised_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)
stacked_plots_all_splits_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id)


