import pickle
import glob
import numpy as np
import torch
import branch_names as na
from functions.classification import get_anomalies, CLASSIFICATION_CHECK
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, mix_quark_gluon_samples, load_trials
from plotting.stacked import *
from plotting.roc import *
from functions.data_manipulation import (
    separate_anomalies_from_regular,
    cut_on_length,
    train_dev_test_split,
)
from plotting.svm_boundary import svm_boundary_plots

# file_name(s) - comment/uncomment when switching between local/Nikhef
#file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny_100k.root"
#file_name = "samples/time_cluster_5k.root"
#file_name = "samples/mixed_1500jets_pct:90g_10q.p"

# use for 90 % gluon 10 % quark ROC curves
job_ids = [
    # "10993304",
    # "10993305",
    "11120653", # hp qg
    "11120654",
    "11120655", 
    "11316965", # hp qg
    "11316966",
    "11316967",
    "11524829", # hp qg - last_reversed only
    "11461549", # reg last
    "11461550",
    "11474168", # reg mean
    "11474169",
    "11478120", # reg last_reversed
    "11478121",
    "11524829", # hp op last_reversed
]


# use for 50 % gluon 50 % quark stacked plots!
# pythia
# job_ids = [
#     "11474168", # reg mean - lowest cost - trial 9"
#     "11474168",  # reg mean - best regtraining - trial 5
#     "11461550", # reg last - best regtraining - trial 7
#     "11478121", # last_reversed - highest auc best regtraining - trial 1
#     '11120653', # hp training mean - highest auc total - trial 11     
# ] 
# trial_nrs = [9, 5, 7, 1, 11] # check auc scores before this
# jewel
# job_ids = [
#     "11542141", # trial 3
# #     # "11542142", # 1
# #     # "11542143", # 8
# #     # "11542143", # 9
# #     # '11542144', # 3
# #     "11542144", # 9
# ] 
# trial_nrs = [6]#3, 1, 8, 9, 3, 9]

out_files = [] # if previously created a specific sample, otherwise leave empty

g_percentage = 90 # for evaluation of stacked plots 50, ROC would be 90
mix = True        # set to true if mixture of q and g is required
kt_cut = None         # for dataset, splittings kt > 1.0 GeV, assign None if not using
save_flag = True
show_distribution_percentages_flag = False
features = [na.recur_jetpt, na.recur_dr, na.recur_z] # features to do analysis on


# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# collect all auc values from ROC curves
all_aucs = {}

### Regular Training options ### TODO make this automatic 
# get dr_cut as in the regular training!
dr_cut = None# np.linspace(0,0.4,len(job_ids)+1)[i+1] 

###--------------------------###

# you can load your premade mix here: pickled file
# Load and filter data for criteria eta and jetpt_cap
# You can load your premade mix here: pickled file w q/g mix

if out_files:
    jets_recur, jets = torch.load(file_name)
elif mix:
    jets_recur, jets, file_name_mixed_sample = mix_quark_gluon_samples(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], g_percentage=g_percentage, kt_cut=kt_cut, dr_cut=dr_cut)
else:
    jets_recur, _ = load_n_filter_data(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)

# split data into (train, val, test) like 70/10/20 if splits are set at [0.7, 0.1]
_, _, split_test_data_recur = train_dev_test_split(jets_recur, split=[0.7, 0.1])
_, _, split_test_data = train_dev_test_split(jets, split=[0.7, 0.1])
# split_test_data_recur = jets_recur
# split_test_data= jets

# # split data into quark and gluon jets
g_jets_recur = split_test_data_recur[split_test_data[na.parton_match_id] == 21]
q_jets_recur = split_test_data_recur[abs(split_test_data[na.parton_match_id]) < 7]
print("Loading data complete")       

for i, job_id in enumerate(job_ids):
    print(f"\nAnomalies run: {i+1}, job_id: {job_id}") # , for dr_cut: {dr_cut}")
    
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(f"No succesful trial for job: {job_id}. Try to complete a new training with same settings.")
        continue
    print("Loading trials complete")
    
    # ROC curves, use 90/10 mixture
    collect_aucs = ROC_curve_qg(g_jets_recur, q_jets_recur, trials, job_id)
    all_aucs[job_id] = collect_aucs
    continue
    
    # stacked plots, use 50/50 mixture
    # gluon jets, get anomalies and normal out
    num = trial_nrs[i] if trial_nrs else range(len(trials))

    type_jets = "g_jets"
    print(f"For {type_jets}")
    _, g_jets_index_tracker, g_classification_tracker = get_anomalies(g_jets_recur, job_id, trials, file_name, jet_info=type_jets)
    g_anomaly, g_normal = separate_anomalies_from_regular(
        anomaly_track=g_classification_tracker[num],
        jets_index=g_jets_index_tracker[num],
        data=g_jets_recur,
    )

    # quark jets, get anomalies and normal out
    type_jets = "q_jets"
    print(f"For {type_jets}")
    _, q_jets_index_tracker, q_classification_tracker = get_anomalies(q_jets_recur, job_id, trials, file_name, jet_info=type_jets)
    q_anomaly, q_normal = separate_anomalies_from_regular(
        anomaly_track=q_classification_tracker[num],
        jets_index=q_jets_index_tracker[num],
        data=q_jets_recur,
    )
    
    # other selection/cuts
    extra_cuts = False
    if extra_cuts == True:
        
        # cut on length
        length = 3
        g_anomaly = cut_on_length(g_anomaly, length, features)
        g_normal = cut_on_length(g_normal, length, features)
        q_anomaly = cut_on_length(q_anomaly, length, features)
        q_normal = cut_on_length(q_normal, length, features)
        job_id += f"_jet_len_{length}"
        
        pass
    
    # all possible stacked plots, comment/uncomment which one you want
    stacked_plots_mean_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    stacked_plots_mean_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    stacked_plots_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    stacked_plots_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    stacked_plots_last_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    stacked_plots_last_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    # stacked_plots_normalised_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    # stacked_plots_normalised_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    #stacked_plots_splittings_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    #stacked_plots_splittings_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    #stacked_plots_all_splits_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
    #stacked_plots_all_splits_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, num)
print(f"All AUC values for these jobs:\n{all_aucs}")
