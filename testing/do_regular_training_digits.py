"""
As do_regular_training.py but made for the digits dataset.

Hyperopt cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a 
Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions 
for Vision Architectures. To appear in Proc. of the 30th International Conference 
on Machine Learning (ICML 2013).
"""

from testing_functions import load_digits_data
from plotting_test import sample_plot
from functions.training import REGULAR_TRAINING, run_full_training

import matplotlib.pyplot as plt
import random

from hyperopt import (
    hp,
)

### --- User input --- ###
# Set hyperspace and variables
runs = 10
max_evals = 10
max_attempts = 8
patience = 10
multicore_flag = False
print_dataset_info = False
save_results_flag = True
plot_flag = True
plot_sample = False
random.seed(0)  # for shuffling of data sequences

# notes onrrun, added to run_info.p, keep short or leave empty
run_notes = "0:0.9 9:0.1[75:150],bs=2000, 4 evals, nu=0.5, mean pool"

# set hyperspace to 1 setting for regular training
space = hp.choice(
    "hyper_parameters",
    [
        {
            "batch_size": hp.choice("num_batch", [2000]),
            "hidden_dim": hp.choice("hidden_dim", [2]),
            "num_layers": hp.choice("num_layers", [1]),
            "min_epochs": hp.choice("min_epochs", [int(150)]),
            "learning_rate": 10 ** hp.choice("learning_rate", [-3]),
            "epsilon": 10 ** hp.choice("epsilon", [-9]),
            "dropout": hp.choice(
                "dropout", [0]
            ),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.5]),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale"]
            ),  # "scale" or "auto"[ 0.23 was the defeault before], auto seems weird
            "scaler_id": hp.choice(
                "scaler_id", ["minmax"]
            ),  # "minmax" = MinMaxScaler or "std" = StandardScaler
            "pooling": hp.choice("pooling", ["mean"]),  # "last" , "mean"
        }
    ],
)

# ---------------------- #


# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"
file_name = train_file + "," + test_file

# get digits data
train_dict = load_digits_data(train_file, print_dataset_info=print_dataset_info)
test_dict = load_digits_data(test_file)

# mix "0" = 90% as normal data with "9" = 10% as anomalous data
train_data = train_dict["0"][:675] + train_dict["9"][75:150]
test_data = test_dict["0"][:360] + test_dict["9"][:40]

# make roc data
zeros = test_dict["0"][:360]
nines = test_dict["9"][:40]
roc_data = [{"data": item, "y_true": 1} for item in zeros] + [
    {"data": item, "y_true": 0} for item in nines
]

# plot sample
if plot_sample:
    index = 49
    plt.figure()
    sample_plot(train_dict["0"], index, label="0")
    sample_plot(train_dict["9"], index, label="9")
    plt.legend()
    plt.show()

# shuffle datasets to make anomalies appear randomly
random.shuffle(train_data)
random.shuffle(test_data)

# do full training on digits data
run_full_training(
    TRAINING_TYPE=REGULAR_TRAINING,
    file_name=file_name,
    space=space,
    train_data=train_data,
    val_data=test_data,
    roc_data=roc_data,
    max_evals=max_evals,
    max_attempts=max_attempts,
    patience=patience,
    multicore_flag=multicore_flag,
    save_results_flag=save_results_flag,
    plot_flag=plot_flag,
    run_notes=run_notes,
)
