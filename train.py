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
from typing import no_type_check

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import uproot
import awkward as ak
import pandas as pd

import names as na
from ai.model import *
from ai.dataset import *
from functions import data_loader

# load root dataset into a pandas dataframe
fileName = "./samples/JetToyHIResultSoftDropSkinny.root"
g_jets, q_jets, g_recur_jets, q_recur_jets = data_loader.load_n_filter_data(fileName)

print(ak.to_pandas(g_jets).head(), len(ak.to_pandas(g_jets)), sep="\n")
print(ak.to_pandas(g_recur_jets).head(), len(ak.to_pandas(g_recur_jets)), sep="\n")

# Recursive gluon jet data in list form
g_list = data_loader.format_ak_to_list(g_recur_jets)

# check if gpu is available, otherwise use cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

batch_size = 64
train_data = JetDataset(g_list[:700])
test_data = JetDataset(g_list[700:1000])
train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)
print("hi")


def train(n_epochs: int, lr: float = 0.04):

    # alg 2 step 1
    # build a model
    model = LSTM(3, 4, 4, 3, device)
    theta_0 = None  # alg 2
    w_0 = None  # alg 2
    rho_0 = None  # alg 2

    # optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # alg 2 step 2
    eps = None  # alg 2

    # alg 2 step 3
    k = -1  # alg  2

    # alg 2 step 4
    for i in range(n_epochs):  # alg 2

        # alg 2 step 5
        k = k + 1  # alg 2

        # alg 2 step 6
        theta_k = None  # alg 2
        h_i_n = None  # using theta_k alg 2 step 6

    return


train(3)
