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
    trials_df_and_minimum,
)
from functions.data_loader import load_n_filter_data
from functions.training import try_hyperparameters
from plotting.general import cost_condition_plots, violin_plots

# from autograd import elementwise_grad as egrad


from hyperopt import (
    fmin,
    tpe,
    hp,
    space_eval,
    STATUS_OK,
    Trials,
    SparkTrials,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
from functools import partial

import torch
import os
import time
import torch

import pandas as pd
import numpy as np

import branch_names as na


### User Input  ###

# file_name(s) - comment/uncomment when switching between local/Nikhef
#file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# set run settings
max_evals = 4
patience = 5
kt_cut = False              # for dataset, splittings kt > 1.0 GeV
debug_flag = True          # for using debug space = only 1 configuration of hp
multicore_flag = False       # for using SparkTrials or Trials
save_results_flag = True    # for saving trials and runtime
plot_flag = True            # for making cost condition plots, only works if save_results_flag is True

# notes on run, added to run_info.p, keep short or leave empty
run_notes = ""

###-------------###


# set hyper space and variables
space = hp.choice(
    "hyper_parameters",
    [
        {  # TODO change to quniform -> larger search space (min, max, stepsize (= called q))
            "batch_size": hp.quniform("num_batch", 300, 1000, 100),
            "hidden_dim": hp.choice("hidden_dim", [6, 9, 12, 20, 50, 100, 200]),
            "num_layers": hp.choice(
                "num_layers", [1]
            ),  # 2 layers geeft vreemde resultaten bij cross check met fake jets, en in violin plots blijkt het niks toe te voegen
            "min_epochs": hp.choice(
                "min_epochs", [int(30)]
            ),  # lijkt niet heel veel te doen
            "learning_rate": 10 ** hp.quniform("learning_rate", -6, -3, 1),
            # "decay_factor": hp.choice("decay_factor", [0.1, 0.4, 0.5, 0.8, 0.9]), #TODO
            "dropout": hp.choice(
                "dropout", [0]
            ),  # voegt niks toe, want we gebuiken één layer, dus dropout niet nodig
            "output_dim": hp.choice("output_dim", [1]),
            "svm_nu": hp.choice("svm_nu", [0.05, 0.001]),  # 0.5 was the default
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
                    #[na.recur_dr, na.recur_jetpt],
                    #[na.recur_dr, na.recur_z],
                    #[na.recur_jetpt, na.recur_z],
                ],
            ),
            "pooling": hp.choice("pooling",["last"]), # "last" , "mean"
        }
    ],
)

# dummy space for debugging
space_debug = hp.choice(
    "hyper_parameters",
    [
        {  # TODO change to quniform -> larger search space (min, max, stepsize (= called q))
            "batch_size": hp.choice("num_batch", [50]),
            "hidden_dim": hp.choice("hidden_dim", [6]),
            "num_layers": hp.choice("num_layers", [1]),
            "min_epochs": hp.choice("min_epochs", [int(25)]),
            "learning_rate": hp.choice("learning_rate", [1e-5]),
            # "decay_factor": hp.choice("decay_factor", [0.1, 0.4, 0.5, 0.8, 0.9]),
            "dropout": hp.choice("dropout", [0]),
            "output_dim": hp.choice("output_dim", [1]),
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
            "pooling": hp.choice("pooling", ["last"]),
        }
    ],
)
# set space if debug
if debug_flag:
    space = space_debug

# start time
start_time = time.time()

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name, kt_cut=kt_cut)
print("Loading data complete")
# split data
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])
print("Splitting data complete")

# set trials or sparktrials
if multicore_flag:
    cores = os.cpu_count() if os.cpu_count() < 10 else 10 
    trials = SparkTrials(
        parallelism=cores
    )  # run as many trials parallel as the nr of cores available
    print(f"Hypertuning {max_evals} evaluations, on {cores} cores:\n")
else:
    trials = Trials()  # NOTE keep for debugging since can't do with spark trials
        
# hyper tuning and evaluation      
best = fmin(
    partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
        try_hyperparameters,
        dev_data=dev_data,
        plot_flag=False,
        patience=patience,
    ),
    space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)
print(f"\nHypertuning completed on dataset:\n{file_name}")

# saving spark_trials as dictionaries
# source https://stackoverflow.com/questions/63599879/can-we-save-the-result-of-the-hyperopt-trials-with-sparktrials
pickling_trials = dict()
for k, v in trials.__dict__.items():
    if not k in ["_spark_context", "_spark"]:
        pickling_trials[k] = v

# collect df and print best models
df, min_val, min_df, parameters = trials_df_and_minimum(pickling_trials, "loss")

# check to save results
if save_results_flag:
    # set out file to job_id for parallel computing
    job_id = os.getenv("PBS_JOBID")
    if job_id:
        job_id = job_id.split('.')[0]
    else:
        job_id = time.strftime('%d_%m_%y_%H%M')
    
    out_file = f"storing_results/trials_test_{job_id}.p"
    
    # save trials as pickling_trials object
    torch.save(pickling_trials, open(out_file, "wb"))

    # check to make plots
    if plot_flag:
        cost_condition_plots(pickling_trials, job_id)
        violin_plots(df, min_val, min_df, parameters, [job_id], "loss")
        print("\nPlotting complete")
    
    # store run info
    run_time = time.time() - start_time
    run_info = f"{job_id}\ton: {file_name}\truntime: {run_time:.2f} s"
    run_info = run_info + f"\tnotes: {run_notes}\n" if run_notes else run_info + "\n"
    with open("storing_results/run_info.p", 'a+') as f:
        f.write(run_info)
    print(f"\nCompleted run in: {run_time}")

    # load torch.load(r"storing_results\trials_test.p",map_location=torch.device('cpu'), pickle_module=pickle)
