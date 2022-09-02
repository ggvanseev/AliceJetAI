"""
Using algorithm 1 of Unsupervised Anomaly Detection With LSTM Neural Networks
Sauce: Tolga Ergen and Suleyman Serdar Kozat, Senior Member, IEEE

-----------------------------------------------------------------------------------------
Algorithm 1: Quadratic Programming-Based Training for the Anomaly Detection Algorithm
             Based on OC-SVM
-----------------------------------------------------------------------------------------
1. Initialize the LSTM parameters as θ_0 and the dual OC-SVM parameters as α_0
2. Determine a threshold ϵ as convergence criterion
3. k = −1
4. do
5.    k = k+1
6.    Using θ_k, obtain {h}^n_{i=1} according to Fig. 2
7.    Find optimal α_{k+1} for {h}^n_{i=1} using (20) and (21)
8.    Based on α_{k+1}, obtain θ_{k+1} using (24) and Remark 3
8. while (κ(θ_{k+1}, α{k+1})− κ(θ_k, α))^2 > ϵ
9. Detect anomalies using (19) evaluated at θ_k and α_k
-----------------------------------------------------------------------------------------

(20): α_1 = 1 − S − α_2, where S= sum^n_{i=3} α_i.
(21): α_{k+1,2} = ((α_{k,1} + α_{k,2})(K_{11} − K_{12})  + M_1 − M_2) / (K_{11} + K_{22}
                                                                               − 2K_{12})
      K_{ij} =def= h ^T_iT h _j, Mi =def= sum^n_{j=3} α_{k,j}K_{ij}
(24): W^(·)_{k+1} = (I + (mu/2)A_k)^-1 (I− (mu/2)A_k) W^(·)_k
      Ak = Gk(W(·))T −W(·)GT

Dual problem of the OC-SVM:
(22): min_{theta}  κ(θ, α_{k+1}) = 1/2 sum^n_{i=1} sum^n_{j=1} α_{k+1,i} α_{k+1,j} h^T_i h_j
(23): s.t.: W(·)^T W(·) = I, R(·)^T R(·) = I and b(·)^T b(·) = 1

Remark 3: For R(·) and b(·), we first compute the gradient of the objective function with
respect to the chosen parameter as in (25). We then obtain Ak according to the chosen
parameter. Using Ak, we update the chosen parameter as in (24).

         ----------------------------------------------------------------------


Using algorithm 2 of Unsupervised Anomaly Detection With LSTM Neural Networks
Sauce: Tolga Ergen and Suleyman Serdar Kozat, Senior Member, IEEE]
"""

from testing_functions import load_digits_data
from plotting_test import sample_plot
from functions.training import REGULAR_TRAINING, run_full_training

import matplotlib.pyplot as plt
import random

from hyperopt import (
    hp,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).


### --- User input --- ###
# Set hyper space and variables
runs = 10
max_evals = 10
max_attempts = 8
patience = 5
multicore_flag = False
print_dataset_info = False
save_results_flag = True
plot_flag = True
plot_sample = False
random.seed(0) # for shuffling of data sequences

# notes onrrun, added to run_info.p, keep short or leave empty
run_notes = "0:0.9 9:0.1[75:150],bs=2000, 4 evals, nu=0.5, mean pool"

# ---------------------- #

# set space
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
            "dropout": hp.choice("dropout", [0]),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.5]),  # 0.5 was the default
            "svm_gamma": hp.choice("svm_gamma", ["scale"]),  #"scale" or "auto"[ 0.23 was the defeault before], auto seems weird
            "scaler_id": hp.choice("scaler_id", ["minmax"]),  # "minmax" = MinMaxScaler or "std" = StandardScaler
            "pooling": hp.choice("pooling", ["mean"]),  # "last" , "mean"
        }
    ],
)

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
#print('Mixed "9": 675 = 90% of normal data with "0": 75 = 10% as anomalous data for a train set of 750 samples')
test_data = test_dict["0"][:360] + test_dict["9"][:40]
#print('Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples')

# plot random sample
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

run_full_training(
    TRAINING_TYPE=REGULAR_TRAINING,
    file_name=file_name,
    space=space,
    train_data=train_data,
    val_data=test_data,
    max_evals=max_evals,
    max_attempts=max_attempts,
    patience=patience,
    multicore_flag=multicore_flag,
    save_results_flag=save_results_flag,
    plot_flag=plot_flag,
    run_notes=run_notes,
)
