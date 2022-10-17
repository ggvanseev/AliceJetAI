import pickle
from hyperopt import (
    hp,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
import awkward as ak
import numpy as np
import torch

import branch_names as na
from functions.data_manipulation import format_ak_to_list, train_dev_test_split
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, mix_quark_gluon_samples

from functions.training import REGULAR_TRAINING, run_full_training

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from plotting.roc import ROC_plot_curve


### ------------------------------- User Input ------------------------------- ###

# file_name(s) - comment/uncomment when switching between local/Nikhef
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny_100k.root"
# file_name = "samples/SDTiny_jewelNR_120_vac-1.root"
# file_name = "samples/SDTiny_jewelNR_120_simple-1.root"
# file_name = "samples/JetToyHIResultSoftDropTiny.root"

# set data sample settings
out_file = ""
mix = True                  # set to true if you want a mixture of quark and gluon jets
g_percentage = 90           # percentage gluon jets of mixture
kt_cut = None

# set fit settings
svm_nu_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # 0.5 was the default
svm_gamma = "auto"  # "scale" or "auto"[ 0.23 was the defeault before], auto seems weird -> this should not do anything!
scaler_id = "minmax" # "minmax" = MinMaxScaler or "std" = StandardScaler
variables = [na.recur_dr, na.recur_jetpt, na.recur_z]
pooling = "mean"  # "mean", "last" , "last_reversed" NOTE latter not "first" to avoid confusion

# set run settings
notes = f"OCSVM fit on {file_name}"  # Small command on run, will be save to save file.

###-----------------------------------------------------------------------------###

    
# set for this specific run
dr_cut = None

# run notes for this run
print(f"\nStarting run, with notes: {notes}")

# Load and filter data for criteria eta and jetpt_cap
# You can load your premade mix here: pickled file w q/g mix
if out_file:
    jets_recur, jets = torch.load(file_name)
elif mix:
    jets_recur, jets, file_name_mixed_sample = mix_quark_gluon_samples(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], g_percentage=g_percentage, kt_cut=kt_cut, dr_cut=dr_cut)
else:
    jets_recur, jets = load_n_filter_data(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
print("Loading data complete")       

# split data into (train, val, test) like 70/10/20 if splits are set at [0.7, 0.1]
split_train_data, _, split_test_data = train_dev_test_split(jets_recur, split=[0.7, 0.1])
_, _, jets = train_dev_test_split(jets, split=[0.7, 0.1])
print("Splitting data complete")

# scale datasets according to training set
if scaler_id == "minmax":
    scaler = MinMaxScaler()
elif scaler_id == "std":
    scaler = StandardScaler()
split_train_data = format_ak_to_list(split_train_data)
split_test_data = format_ak_to_list(split_test_data)
scaler.fit([s for jet in split_train_data for s in jet]) # fit as single branch

# collect all auc values from ROC curves
all_aucs = {}

for pooling in ["mean", "last", "last_reversed"]:
    train_data = [scaler.transform(d) for d in split_train_data] # then transform it
    test_data = [scaler.transform(d) for d in split_test_data] # then transform it

    # pool data
    if pooling == "mean":
        train_data = [np.mean(jet, axis=0) for jet in train_data]
        test_data = [np.mean(jet, axis=0) for jet in test_data]
    elif pooling == "last":
        train_data = [train_data[i][-1] for i in range(len(train_data))]
        test_data = [test_data[i][-1] for i in range(len(test_data))]
    elif pooling == "last_reversed":
        train_data = [train_data[i][0] for i in range(len(train_data))]
        test_data = [test_data[i][0] for i in range(len(test_data))]
    else:
        print("pooling is wrong!")

    for svm_nu in svm_nu_values:
        # make and fit model
        svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="linear")
        svm_model.fit(train_data)
        alphas = np.abs(svm_model.dual_coef_)[0]

        alphas = alphas / np.sum(alphas)  # NOTE: equation 14, sum alphas = 1

        a_idx = svm_model.support_

        # make predictions and get probabilities for ROC
        classification = svm_model.predict(test_data)
        y_predict = svm_model.decision_function(test_data) # = y_predict

        # count anomalies
        n_anomaly = np.count_nonzero(classification == -1)
        fraction_anomaly = n_anomaly / len(classification)
        print(f"Fraction of {fraction_anomaly:.4f} anomalies found in dataset")

        # get y_true
        y_true = [1 if jet[na.parton_match_id] == 21 else 0 for jet in jets]

        # plot setup
        plot_title = r"ROC Curve of OCSVM With $\nu = {}$".format(svm_nu)
        out_file =  f"testing/output/ROC_OCSVM/nu{svm_nu}_{pooling}"

        # make ROC curve & store for later
        fig, roc_auc = ROC_plot_curve(y_true, y_predict, plot_title, out_file,  xlabel="Normal Fraction Quarks", ylabel="Normal Fraction Gluons")
        #pickle.dump(fig, open(out_file+'.p'), 'wb')
        all_aucs[str(pooling)+"_nu"+str(svm_nu)] = roc_auc
print(f"All AUC values for these jobs:\n{all_aucs}")