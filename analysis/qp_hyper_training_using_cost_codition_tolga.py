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
# from sklearn.externals import joblib
# import joblib

from functions.data_manipulation import (
    train_dev_test_split,
    format_ak_to_list,
)
from functions.data_loader import load_n_filter_data
from functions.training import try_hyperparameters

# from autograd import elementwise_grad as egrad


from hyperopt import (
    fmin,
    tpe,
    hp,
    space_eval,
    STATUS_OK,
    Trials,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
from functools import partial

import torch
import os
import time

import pandas as pd
import numpy as np

# Set hyper space and variables
max_evals = 1
patience = 5
space = hp.choice(
    "hyper_parameters",
    [
        {  # TODO change to quniform -> larger search space (min, max, stepsize (= called q))
            "batch_size": hp.choice("num_batch", [50, 100, 150, 200]),
            "hidden_dim": hp.quniform("hidden_dim", 2, 20, 3),
            "num_layers": hp.choice("num_layers", [1, 2]),
            "min_epochs": hp.choice("min_epochs", [int(5), int(10), int(20)]),
            "learning_rate": hp.choice(
                "learning_rate", [1e-3, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            ),
            "decay_factor": hp.choice("decay_factor", [0.1, 0.4, 0.5, 0.8, 0.9]),
            "dropout": hp.choice("dropout", [0, 0.2, 0.4, 0.6]),
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.05]),  # 0.5 was the default
            "svm_gamma": hp.choice(
                "svm_gamma", ["scale", "auto"]  # Auto seems to give weird results
            ),  # , "scale", , "auto"[ 0.23 was the defeault before]
        }
    ],
)

# file_name(s) - comment/uncomment when switching between local/Nikhef
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# start time
start_time = time.time()

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)
print("Loaded data")

# split data
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])
print("Split data")

trials = Trials()
best = fmin(
    partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
        try_hyperparameters, dev_data=dev_data, plot_flag=False, patience=patience,
    ),
    space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)
print(space_eval(space, best))

# set out file to job_id for parallel computing
job_id = os.getenv("PBS_JOBID")
if job_id:
    out_file = f"storing_results/trials_test_{job_id}.p"
else:
    out_file = f"storing_results/trials_test_{time.strftime('%x_%X')}.p"

torch.save(trials, open(out_file, "wb"))

run_time = pd.DataFrame(np.array([time.time() - start_time]))

run_time.to_csv("storing_results/runtime.p")
# load trials_test = pickle.load(open("/storing_results/trials_test.p", "rb"))
