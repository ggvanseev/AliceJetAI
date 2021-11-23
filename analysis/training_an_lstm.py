"""
Training the LSTM.

Using algorithm 2 of Unsupervised Anomaly Detection With LSTM Neural Networks
Sauce: Tolga Ergen and Suleyman Serdar Kozat, Senior Member, IEEE

-----------------------------------------------------------------------------------------
Algorithm 2: Gradient-Based Training for the Anomaly Detection Algorithm Based on OC-SVM
-----------------------------------------------------------------------------------------
1. Initialize the LSTM parameters as θ_0 and the OC-SVM parameters as w_0 and ρ_0
2. Determine a threshold ϵ as convergence criterion
3. k = −1
4. do
5.    k = k+1
6.    Using θ_k, obtain {h}^n_{i=1} according to Fig. 2
7.    Obtain w_{k+1}, ρ_{k+1} and θ_{k+1} using (33), (35), (37) and Remark 4
8. while (F_τ(w_{k+1},ρ_{k+1},θ_{k+1})− F_τ(w_k,ρ_k,θ_k))^2 > ϵ
9. Detect anomalies using (10) evaluated at w_k, ρ_k and θ_k
-----------------------------------------------------------------------------------------

(33): w_{k+1} = w_k − μ ∇ wF_τ(w,ρ,θ) | w=w_k, ρ=ρ_k, θ=θ_k
(35): ρ_{k+1} = ρ_k − μ ∂F_τ(w,ρ,θ)/(∂p)| w=w_k, ρ=ρ_k, θ=θ_k
(37): W^{(·)}_{k+1}= (I + μ/2 B_k)^{-1} (I− μ/2 B_k) W^{(·)}_k
      B_k = M_k(W^{(·)}_k)^T − W^{(·)}_k M_k^T and 
      Mij =def=  ∂Fτ(w,ρ,θ) / ( ∂W^{(·)}_{ij} )

Objective function of the SVM
F_τ(w , ρ, θ) =def= ||w||^2/2 + 1/(nλ) sum^n_{i=1} S_τ( β_{w ,ρ}(hbar_i) ) − ρ 

Proposition 1: As τ increases, Sτ (βw,ρ (h ̄ i )) uniformly converges to G(βw,ρ(h ̄i)).
As a consequence, our approximation Fτ(w,ρ,θ) converges to the SVM objective function
F(w,ρ,θ) ... Proof of Proposition 1: The proof of the proposition is given in Appendix A.

Remark 4: For R(·) and b(·), we first compute the gradient
of the objective function with respect to the chosen parameter
as in (38). We then obtain Bk according to the chosen
parameter. Using B , we update the chosen parameter as k
in (37).

"""
import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

from functions.data_saver import save_results, save_loss_plots, DataTrackerTrials
from functions.data_manipulation import (
    train_validation_split,
    format_ak_to_list,
)
from functions.training import train_model_with_hyper_parameters, getBestModelfromTrials
from functions.data_loader import load_n_filter_data


from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials

import names as na
from ai.model import LSTM
from functions import data_loader
from functools import partial


# Variables
# File
file_name = "samples\JetToyHIResultSoftDropSkinny.root"

# Output dimension
output_dim = 3

# flags
flag_save_intermediate_results = False
flag_save_loss_plots = False
# hyper tuning space
max_evals = 1
space = hp.choice(
    "hyper_parameters",
    [
        {
            "num_batch": hp.choice("num_batch", [100, 1000]),
            "num_epochs": hp.choice("num_epochs", [1, 2]),
            "num_layers": hp.choice("num_layers", [1]),
            "hidden_size0": hp.choice("hidden_size0", [8]),
            "hidden_size1": hp.choice("hidden_size1", [4]),
            "learning_rate": hp.choice("learning_rate", [0.01]),
            "decay_factor": hp.choice("decay_factor", [0.9]),
            "loss_func": hp.choice("loss_func", ["mse"]),
        }
    ],
)

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)


# only use g_recur_jets
training_data, validation_data = train_validation_split(g_recur_jets, split=0.8)


# Setup datatracker for trials
data_tracker = DataTrackerTrials

# Find best model
trials = Trials()
best = fmin(
    partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
        train_model_with_hyper_parameters,
        training_data=training_data,
        validation_data=validation_data,
        data_tracker=data_tracker,
        output_dim=output_dim,
    ),
    space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)
print(space_eval(space, best))
