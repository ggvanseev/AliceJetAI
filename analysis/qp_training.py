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
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib
# import joblib

import torch

from functions.data_manipulation import (
    train_dev_test_split,
    format_ak_to_list,
    branch_filler,
    lstm_data_prep,
    branch_filler_jit,
)
from functions.data_loader import load_n_filter_data
from functions.training import training_algorithm

from ai.model_lstm import LSTMModel

# from autograd import elementwise_grad as egrad

from plotting.svm_boundary import plot_svm_boundary_2d
from functions.validation import validation_distance_nu


file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# Variables:
batch_size = 200
output_dim = 1
hidden_dim = 18
layer_dim = 1
dropout = 0.2
min_epochs = 100
max_epochs = 5000
learning_rate = 1e-5
weight_decay = 1e-10
nu = 0.01

eps = 1e-8  # test value for convergence
patience = 5  # value for number of epochs before stopping after seemingly convergence

# Load and filter data for criteria eta and jetpt_cap
_, _, g_recur_jets, _ = load_n_filter_data(file_name)
g_recur_jets = format_ak_to_list(g_recur_jets)

# split data
train_data, dev_data, test_data = train_dev_test_split(g_recur_jets, split=[0.8, 0.1])
batch_size = len(train_data)

train_data, track_jets_train_data = branch_filler_jit(train_data, batch_size=batch_size)
dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=100)

# Note this has to be saved with the model, to ensure data has the same form.
# Only use train and dev data for now
scaler = (
    MinMaxScaler()
)  # Note this has to be saved with the model, to ensure data has the same form.
train_loader = lstm_data_prep(
    data=train_data, scaler=scaler, batch_size=batch_size, fit_flag=True
)
dev_loader = lstm_data_prep(data=dev_data, scaler=scaler, batch_size=100)

input_dim = len(train_data[0])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_params = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "output_dim": output_dim,
    "dropout_prob": dropout,
    "batch_size": batch_size,
    "device": device,
}

training_params = {
    "min_epochs": min_epochs,
    "max_epochs": max_epochs,
    "learning_rate": learning_rate,
    "epsilon": eps,
    "patience": patience,
}

# lstm model
lstm_model = LSTMModel(**model_params)

# svm model
svm_model = OneClassSVM(nu=nu, gamma="scale", kernel="rbf")

# path for model - only used for saving
# model_path = f'models/{lstm_model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


### ALGORITHM START ###
lstm_model, svm_model, track_cost, track_cost_condition, _ = training_algorithm(
    lstm_model,
    svm_model,
    train_loader,
    track_jets_dev_data,
    model_params,
    training_params,
    device,
)

# Plotting can be moved later to a seperate map (here for speed)
for x_batch, y_batch in train_loader:
    x_batch = x_batch.view(
        [batch_size, -1, model_params["input_dim"]]
    )  # TODO to device? I am assuming no since we only look at a single datapoint (jet)

hn_reduced, theta_reduced, theta_gradients_reduced = lstm_model(
    x_batch
)  # Pretend as if first jet is 5 splits long, to check if it predicts. This will give an error (TODO)

# only take final part of jet, as in training
hn_reduced = hn_reduced[:, track_jets_dev_data[-1]]


hn_reduced = hn_reduced[-1].detach().numpy()  # only final layer
hn_predicted = svm_model.predict(hn_reduced)

plot_svm_boundary_2d(h_bar=hn_reduced, h_predicted=hn_predicted, svm_model=svm_model)

diff_val = validation_distance_nu(
    nu,
    dev_loader,
    track_jets_dev_data,
    input_dim,
    lstm_model,
    svm_model,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)


diff_train = validation_distance_nu(
    nu,
    train_loader,
    track_jets_train_data,
    input_dim,
    lstm_model,
    svm_model,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)


print("Done")
