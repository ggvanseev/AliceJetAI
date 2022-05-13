"""
Run a Hyper Tuning with Quadratic Programming-Based Training for the Anomaly Detection 
Algorithm Based on OC-SVM with an LSTM with Algorithm 1 of Tolga Ergen's paper
"Unsupervised Anomaly Detection With LSTM Neural Networks.
For reference on the training procedure, see 'functions/training.py'.

Hyper tuning implies a training of LSTM and OC-SVM models on hyper parameters within
a pre-defined space. The best hyper parameters will be selected after training is
completed on a number of trials equal to 'max_evals'.
"""
from functions.training import run_full_training, HYPER_TRAINING
from functions.data_manipulation import train_dev_test_split
from functions.data_loader import load_n_filter_data

from hyperopt import (
    hp,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).

import branch_names as na

### ------------------------------- User Input ------------------------------- ###

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/time_cluster_jewel_5k.root"

# set run settings
max_evals = 1
patience = 5
kt_cut = None  # for dataset, splittings kt > 1.0 GeV, assign None if not using
debug_flag = True  # for using debug space = only 1 configuration of hp
multicore_flag = True  # for using SparkTrials or Trials, turns of when debugging
save_results_flag = True  # for saving trials and runtime
plot_flag = (
    False  # for making cost condition plots, only works if save_results_flag is True
)

run_notes = "Run 7, same as run 6, but lower number of trials per job given time needed."  # Small comment on run, will be saved to save file.

###-----------------------------------------------------------------------------###


# set hyper space and variables
space = hp.choice(
    "hyper_parameters",
    [
        {  # TODO change to quniform -> larger search space (min, max, stepsize (= called q))
            "batch_size": hp.quniform("num_batch", 500, 1000, 100),
            "hidden_dim": hp.choice("hidden_dim", [9, 12, 50, 125, 250, 500]),
            "num_layers": hp.choice(
                "num_layers", [1]
            ),  # 2 layers geeft vreemde resultaten bij cross check met fake jets, en in violin plots blijkt het niks toe te voegen
            "min_epochs": hp.choice(
                "min_epochs", [int(80)]
            ),  # lijkt niet heel veel te doen
            "learning_rate": 10 ** hp.quniform("learning_rate", -6, -3, 1),
            # "decay_factor": hp.choice("decay_factor", [0.1, 0.4, 0.5, 0.8, 0.9]), #TODO
            "dropout": hp.choice(
                "dropout", [0]
            ),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.5, 0.3, 0.2, 0.1]),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale", "auto"]  # Auto seems to give weird results
            ),  # , "scale", , "auto"[ 0.23 was the defeault before]
            "scaler_id": hp.choice(
                "scaler_id", ["minmax", "std"]
            ),  # MinMaxScaler or StandardScaler
            "variables": hp.choice(
                "variables",
                [
                    [na.recur_dr, na.recur_jetpt, na.recur_z],
                    # [na.recur_dr, na.recur_jetpt],
                    [na.recur_dr, na.recur_z],
                    # [na.recur_jetpt, na.recur_z],
                    [
                        na.recur_dr,
                        na.recur_jetpt,
                        na.recur_z,
                        na.recur_tf,
                    ],
                    [
                        na.recur_dr,
                        na.recur_z,
                        na.recur_tf,
                    ],
                    [
                        na.recur_tf,
                        na.recur_jetpt,
                    ],
                ],
            ),
            "pooling": hp.choice("pooling", ["mean", "last"]),  # "last" , "mean"
        }
    ],
)

# dummy space for debugging
space_debug = hp.choice(
    "hyper_parameters",
    [
        {  # TODO change to quniform -> larger search space (min, max, stepsize (= called q))
            "batch_size": hp.choice("num_batch", [20]),
            "hidden_dim": hp.choice("hidden_dim", [6]),
            "num_layers": hp.choice("num_layers", [1]),
            "min_epochs": hp.choice("min_epochs", [int(25)]),
            "learning_rate": hp.choice("learning_rate", [1e-3]),
            # "decay_factor": hp.choice("decay_factor", [0.1, 0.4, 0.5, 0.8, 0.9]),
            "dropout": hp.choice("dropout", [0]),
            "output_dim": hp.choice("output_dim", [2]),
            "svm_nu": hp.choice("svm_nu", [0.9]),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale"]  # Auto seems to give weird results
            ),  # , "scale", , "auto"[ 0.23 was the defeault before]
            "scaler_id": hp.choice("scaler_id", ["std"]),  # minmax or std
            "variables": hp.choice(
                "variables",
                [
                    [na.recur_dr, na.recur_jetpt, na.recur_z],
                    # [na.recur_dr, na.recur_jetpt],
                    # [na.recur_dr, na.recur_z],
                    # [na.recur_jetpt, na.recur_z],
                ],
            ),
            "pooling": hp.choice("pooling", ["mean"]),
        }
    ],
)
# set space if debug
if debug_flag:
    space = space_debug
    multicore_flag = False

# Load and filter data for criteria eta and jetpt_cap
jets_recur, _ = load_n_filter_data(file_name, kt_cut=kt_cut)
print("Loading data complete")

# split data
_, split_dev_data, _ = train_dev_test_split(jets_recur, split=[0.8, 0.1])
print("Splitting data complete")

# do full training
run_full_training(
    TRAINING_TYPE=HYPER_TRAINING,
    file_name=file_name,
    space=space,
    train_data=split_dev_data,
    max_evals=max_evals,
    patience=patience,
    kt_cut=kt_cut,  # for dataset, splittings kt > 1.0 GeV
    multicore_flag=multicore_flag,
    save_results_flag=save_results_flag,
    plot_flag=plot_flag,
    run_notes=run_notes,
)
