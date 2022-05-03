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

import branch_names as na
from functions.data_manipulation import train_dev_test_split
from functions.data_loader import load_n_filter_data, load_n_filter_data_single

from functions.training import REGULAR_TRAINING, run_full_training


### User Input  ###

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
#file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# JEWEL
#file_name = "samples/SDTiny_jewelNR_120_vac-1.root"
#file_name = "samples/SDTiny_jewelNR_120_simple-1.root"
file_name = "samples/JetToyHIResultSoftDropTiny.root"


# set run settings
max_evals = 4
patience = 10
kt_cut = None  # for dataset, splittings kt > 1.0 GeV, assign None if not using
multicore_flag = False  # for using SparkTrials or Trials
save_results_flag = True  # for saving trials and runtime
plot_flag = (
    True  # for making cost condition plots, only works if save_results_flag is True
)

run_notes = "qg with new settings, check if it improves"  # Small command on run, will be save to save file.

###-------------###

# set space
space = hp.choice(
    "hyper_parameters",
    [
        {  
            "batch_size": hp.choice("num_batch", [500]),
            "hidden_dim": hp.choice("hidden_dim", [20]),
            "num_layers": hp.choice("num_layers", [1]),
            "min_epochs": hp.choice("min_epochs", [int(100)]),
            "learning_rate": 10 ** hp.choice("learning_rate", [-4]),
            "dropout": hp.choice("dropout", [0]),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.1]),  # 0.5 was the default
            "svm_gamma": hp.choice("svm_gamma", ["scale"]),  #"scale" or "auto"[ 0.23 was the defeault before], auto seems weird
            "scaler_id": hp.choice("scaler_id", ["minmax"]),  # "minmax" = MinMaxScaler or "std" = StandardScaler
            "variables": hp.choice("variables",[[na.recur_dr, na.recur_jetpt, na.recur_z]]),
            "pooling": hp.choice("pooling", ["last"]),  # "last" , "mean"
        }
    ],
)

# Load and filter data for criteria eta and jetpt_cap
g_recur_jets, q_recur_jets = load_n_filter_data_single(file_name, kt_cut=kt_cut)
print("Loading data complete")

# Mix sample with e.g. 90% gluons and 10% quarks
mixed_sample = ak.concatenate((g_recur_jets[:1350],q_recur_jets[:150]))
# TODO first shuffle mixed_sample? nah, it's not really possible within awkward, you'd have to get it out first

# remove from memory
del g_recur_jets, q_recur_jets

# split data
split_train_data, _, split_val_data = train_dev_test_split(
    mixed_sample, split=[0.7, 0.1]
)
print("Splitting data complete")

# do full training
run_full_training(
    TRAINING_TYPE=REGULAR_TRAINING,
    file_name=file_name,
    space=space,
    train_data = mixed_sample,
    val_data= split_val_data,
    max_evals=max_evals,
    patience=patience,
    kt_cut=kt_cut,  # for dataset, splittings kt > 1.0 GeV
    multicore_flag=multicore_flag,
    save_results_flag=save_results_flag,
    plot_flag=plot_flag,
    run_notes=run_notes,
)