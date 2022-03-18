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
from functions.data_loader import load_digits_data
from functions.training import REGULAR_TRAINING
from plotting.general import cost_condition_plot

import pickle
import os
import time
import torch
import scipy.io
import h5py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### --- User input --- ###
# Set hyper space and variables
max_evals = 10
patience = 10
print_dataset_info = True
save_results_flag = True
plot_flag = True

# notes on run, added to run_info.p, keep short or leave empty
run_notes = "first succesful test with digits"

# set hyper parameters
hyper_parameters = dict()
hyper_parameters["batch_size"] = 500
hyper_parameters["output_dim"] = 2
hyper_parameters["hidden_dim"] = 2
hyper_parameters["num_layers"] = 1
hyper_parameters["dropout"] = 0
hyper_parameters["min_epochs"] = 25
hyper_parameters["learning_rate"] = 5e-5
hyper_parameters["svm_nu"] = 0.5
hyper_parameters["svm_gamma"] = "scale"
hyper_parameters["scaler_id"] = "minmax"
hyper_parameters["pooling"] = "mean"

# ---------------------- #


### --- Program Start --- ###

# start time
start_time = time.time()

# storing dict:
trials = dict()

# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"

# get digits data
train_dict = load_digits_data(train_file, print_dataset_info=print_dataset_info)
test_dict = load_digits_data(test_file)

# plot random sample
# plt.figure()
# plt.scatter(*train_dict["8"][24].T)
# plt.xlim(0,500)
# plt.ylim(0,500)
# plt.show()

# mix "0" = 90% as normal data with "9" = 10% as anomalous data
train_data = train_dict["0"][:675] + train_dict["9"][:75]
print(
    'Mixed "0": 675 = 90% of normal data with "9": 75 = 10% as anomalous data for a train set of 750 samples'
)
test_data = test_dict["0"][:360] + test_dict["9"][:40]
print(
    'Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples'
)


# track distance_nu
distance_nu = []

training = REGULAR_TRAINING()

for trial in range(max_evals):
    trials[trial] = training.run_training(
        hyper_parameters=hyper_parameters,
        train_data=train_data,
        val_data=test_data,
        patience=patience,
    )

    distance_nu.append(trials[trial]["loss"])

    print(f"Best distance so far is: {min(distance_nu)}")


# saving spark_trials as dictionaries
# source https://stackoverflow.com/questions/63599879/can-we-save-the-result-of-the-hyperopt-trials-with-sparktrials
# pickling_trials = dict()
# for k, v in trials.__dict__.items():
#     if not k in ["_spark_context", "_spark"]:
#         pickling_trials[k] = v

# collect df and print best models
# df, min_val, min_df, parameters = trials_df_and_minimum(trials, "loss")

# check to save results
if save_results_flag:
    # set out file to job_id for parallel computing
    job_id = os.getenv("PBS_JOBID")
    if job_id:
        job_id = job_id.split(".")[0]
    else:
        job_id = time.strftime("%d_%m_%y_%H%M")

    out_file = f"storing_results/trials_test_{job_id}.p"

    # save trials as pickling_trials object
    torch.save(trials, open(out_file, "wb"))

    # check to make plots
    if plot_flag:
        # make out directory if it does not exist yet
        out_dir = f"output/cost_condition_{job_id}"
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
        for trial in trials:
            fig = cost_condition_plot(trials[trial], job_id)
            fig.savefig(out_dir + "/" f"trial_{trial}.png")
        # violin_plots(df, min_val, min_df, parameters, [job_id], "loss")
        print("\nPlotting complete")

    # store run info
    run_time = time.time() - start_time
    run_info = f"{job_id}\ton: {train_file}\truntime: {run_time:.2f} s"
    run_info = run_info + f"\tnotes: {run_notes}\n" if run_notes else run_info + "\n"
    with open("storing_results/run_info.p", "a+") as f:
        f.write(run_info)
    print(f"\nCompleted run in: {run_time}")

    # load torch.load(r"storing_results\trials_test.p",map_location=torch.device('cpu'), pickle_module=pickle)
