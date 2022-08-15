"""
Run a Regular training with Quadratic Programming-Based Training for the Anomaly Detection 
Algorithm Based on OC-SVM with an LSTM with Algorithm 1 of Tolga Ergen's paper
"Unsupervised Anomaly Detection With LSTM Neural Networks.
For reference on the training procedure, see 'functions/training.py'.

Regular training implies a training of LSTM and OC-SVM models on a fixed set of
hyper parameters. 'max_evals' determines the number of training sessions that
will be done, in order to obtain different results or to make sure that at least
one training will be successful.
"""
from hyperopt import (
    hp,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
import awkward as ak
import numpy as np
import torch

import branch_names as na
from functions.data_manipulation import train_dev_test_split
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, mix_quark_gluon_samples

from functions.training import REGULAR_TRAINING, run_full_training


### ------------------------------- User Input ------------------------------- ###

# file_name(s) - comment/uncomment when switching between local/Nikhef
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
# file_name = "samples/JetToyHIResultSoftDropSkinny.root"
# file_name = "samples/SDTiny_jewelNR_120_vac-1.root"
# file_name = "samples/SDTiny_jewelNR_120_simple-1.root"
# file_name = "samples/JetToyHIResultSoftDropTiny.root"

# set data sample settings
out_file = ""               # if previously created a specific sample, otherwise leave empty
mix = True                  # set to true if you want a mixture of quark and gluon jets
g_percentage = 90           # percentage gluon jets of mixture

# set run settings
max_evals = 1               # nr. of trials with same settings
max_attempts = 8            # nr. of times algorithm will retry upon failed training
patience = 10               # nr. of epochs to run after cost condition is met
kt_cut = None               # for dataset, splittings kt > 1.0 GeV, assign None if not using
multicore_flag = False      # for using SparkTrials or Trials, turn of for debuging
save_results_flag = True    # for saving trials and runtime
plot_flag = (
    True                    # for making cost condition plots, only works if save_results_flag is True
)

train_notes = "regular training mixed 90g 10q, last_pooled, lr=e-3, nu = 0.1, hidden dim = 3"  # Small command on run, will be save to save file.

###-----------------------------------------------------------------------------###


# set multiple runs
runs = 4
for i in range(runs):
    
    # set for this specific run
    dr_cut = None
    
    # run notes for this run
    run_notes = train_notes + f", run {i+1}/{runs}"  # Small command on run, will be save to save file.
    print(f"\nStarting run: {i}, with notes: {run_notes}")
    
    # set space
    space = hp.choice(
        "hyper_parameters",
        [
            {
                "batch_size": hp.choice("num_batch", [300]),
                "hidden_dim": hp.choice("hidden_dim", [3]),
                "num_layers": hp.choice("num_layers", [1]),
                "min_epochs": hp.choice("min_epochs", [int(100)]),
                "learning_rate": 10 ** hp.choice("learning_rate", [-3]),
                "dropout": hp.choice(
                    "dropout", [0]
                ),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
                "output_dim": hp.choice("output_dim", [1]),
                "svm_nu": hp.choice("svm_nu", [0.1]),  # 0.5 was the default
                "svm_gamma": hp.choice(
                    "svm_gamma", ["auto"]
                ),  # "scale" or "auto"[ 0.23 was the defeault before], auto seems weird
                "scaler_id": hp.choice(
                    "scaler_id", ["minmax"]
                ),  # "minmax" = MinMaxScaler or "std" = StandardScaler
                "variables": hp.choice(
                    "variables", [[na.recur_dr, na.recur_jetpt, na.recur_z]]
                ),
                "pooling": hp.choice("pooling", ["mean"]),  # "last" , "mean"
            }
        ],
    )

    # Load and filter data for criteria eta and jetpt_cap
    # You can load your premade mix here: pickled file w q/g mix
    if out_file:
        jets_recur, jets = torch.load(file_name)
    elif mix:
        jets_recur, jets, file_name_mixed_sample = mix_quark_gluon_samples(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], g_percentage=g_percentage, kt_cut=kt_cut, dr_cut=dr_cut)
    else:
        jets_recur, _ = load_n_filter_data(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
    print("Loading data complete")       

    # split data
    split_train_data, _, split_val_data = train_dev_test_split(jets_recur, split=[0.7, 0.1])
    print("Splitting data complete")

    # do full training
    run_full_training(
        TRAINING_TYPE=REGULAR_TRAINING,
        file_name=file_name_mixed_sample,
        space=space,
        train_data=split_train_data,
        val_data=split_val_data,
        max_evals=max_evals,
        max_attempts=max_attempts,
        patience=patience,
        kt_cut=kt_cut,  # for dataset, splittings kt > 1.0 GeV
        multicore_flag=multicore_flag,
        save_results_flag=save_results_flag,
        plot_flag=plot_flag,
        run_notes=run_notes,
    )
