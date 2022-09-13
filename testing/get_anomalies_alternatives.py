"""
In this file I create a few different ways to select jets as normal or anomalies judging by
methods discussed with Marco. We wanted to compare the selective powers of the lstm and, 
most notably, the ocsvm with a more naive method of selecting anomalous jets.

The naive method involves placing cuts on the first SoftDrop splitting for one or
several variables. The most important variable will be the R_g angle of the split, but
I will also try the method for the z fraction of momentum. The cut will first be applied
on a low value of the variable, every jet excluded (or included) by the cut will be con-
sidered anomalous. The result of anomalous vs normal will then be compared with their
parton PDG to see whether the anomalous jets are indeed quark jets or gluon jets. This
then gives true positives and a false positives. Afterwards, cuts will be placed for
increasingly higher values and tested against the true label of quark vs gluon to obtain
the true postiive rate and false positive rate, from which a ROC curve can be adopted.

This is then used for the purpose of comparing three different tests:
1. Cuts on a normal dataset not using the lstm / ocsvm
2. Cuts on the hidden states of the lstm, not using the ocsvm. 
    (Use only hidden dim of 3, test all 3 dims)
3. Using the full model of the lstm with the ocsvm as done previously.

"""

import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import awkward as ak
import os

import branch_names as na
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, load_trials, mix_quark_gluon_samples
from functions.data_manipulation import (
    separate_anomalies_from_regular,
    cut_on_length,
    train_dev_test_split,
)
from plotting.roc import *

#file_name(s) - comment/uncomment when switching between local/Nikhef
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
#file_name = "samples/time_cluster_5k.root"
#file_name = "samples/mixed_1500jets_pct:90g_10q.p"

g_percentage = 90
num = 0 # trial nr.
save_flag = True
show_distribution_percentages_flag = False

pre_made_mix = False    # created in regular training
mixed = True            # set to true if you just want the complete sample and not quark/gluon split
qg = False
kt_cut = None           # for dataset, use splittings kt > kt_cut, assign None if not using
dr_cut = None           # for dataset, use splittings dr > dr_cut, assign None if not using

# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if pre_made_mix:
        jets_recur, jets = torch.load(file_name, map_location=device)
elif mixed:
    # Load and filter data for criteria eta and jetpt_cap
    jets_recur, jets, file_name = mix_quark_gluon_samples(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
elif qg:
    g_jets_recur, q_jets_recur, _, _ = load_n_filter_data_qg(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
else:
    jets_recur, jets = load_n_filter_data(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
    
# split data TODO see if it works -> test set too small for small dataset!!! -> using full set
# split data TODO see if it works -> test set too small for small dataset!!! -> using full set
_, split_test_data_recur, _ = train_dev_test_split(jets_recur, split=[0.7, 0.1])
_, split_test_data, _ = train_dev_test_split(jets, split=[0.7, 0.1])
# split_test_data_recur = jets_recur
# split_test_data= jets


# # split data into quark and gluon jets
g_jets_recur = split_test_data_recur[split_test_data[na.parton_match_id] == 21]
q_jets_recur = split_test_data_recur[abs(split_test_data[na.parton_match_id]) < 7]


# Test hand cuts -> results in ROC curves and printout of AUC TODO put AUC in ROC curve
ROC_anomalies_hand_cut(g_jets_recur, q_jets_recur, na.recur_dr)
ROC_anomalies_hand_cut(g_jets_recur, q_jets_recur, na.recur_z)
ROC_anomalies_hand_cut(g_jets_recur, q_jets_recur, na.recur_jetpt)

# Test hand cuts from LSTM results
job_ids = [
    "22_08_09_1909",
    "22_08_09_1934",
    "22_08_09_1939",
    "22_08_09_1941",
    "22_08_11_1520",
    "10993304",
    "10993305",
    "11120653",   
    "11120654",
    "11120655",
]
job_ids = [
    "11461549",
    "11461550",
]
#job_ids = [
#    "10993304",
#    "10993305",
#]

# collect all auc values from ROC curves
all_aucs = {}

for job_id in job_ids:
    
    print(f"\nFor job: {job_id}")
    
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(f"No succesful trial for job: {job_id}. Try to complete a new training with same settings.")
        continue
    print("Loading trials complete")
    
    collect_aucs = ROC_anomalies_hand_cut_lstm(g_jets_recur, q_jets_recur, job_id, trials)
    all_aucs[job_id] = collect_aucs
