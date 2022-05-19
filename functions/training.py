"""
This file is structured to contain all required functions and classes for 
the subsequent training of the LSTM and OC-SVM.

The training is based on the Quadratic Programming-Based Training algorithm 
described in Tolga Ergen's paper "Unsupervised Anomaly Detection With LSTM Neural
Networks. The algorithm is given below in the THEORY part.

The file contains the training algorithm, the TRAINING class that is used to invoke
training of the models on the data, and two child classes of TRAINING:
    - REGULAR_TRAINING: Class used when a regular training is to be done using
    a set space of hyper parameter values, for n number of evaluations
    - HYPER_TRAINING: Class used when a hyper tuning of the model is to be done
    to tune the model/training parameters for this dataset
The difference between the two is mainly in which part of the data it uses:
train, test and development.
Lastly, run_full_training is a function that orchestrates one training session. 


         ------------------------------- THEORY ---------------------------------
Algorithm 1 of Unsupervised Anomaly Detection With LSTM Neural Networks
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
"""

import numpy as np
import time
import os
from copy import copy
import torch
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from hyperopt import (
    fmin,
    tpe,
    STATUS_OK,
    STATUS_FAIL,
    Trials,
    SparkTrials,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
from functools import partial


from functions.data_manipulation import (
    branch_filler,
    lstm_data_prep,
    h_bar_list_to_numpy,
    scaled_epsilon_n_max_epochs,
    format_ak_to_list,
    shuffle_batches,
    trials_df_and_minimum,
)
from functions.optimization_orthogonality_constraints import (
    kappa,
    optimization,
)
from functions.run_lstm import calc_lstm_results
from functions.validation import calc_percentage_anomalies

from ai.model_lstm import LSTMModel

from plotting.cost_condition import cost_condition_plots
from plotting.svm_boundary import svm_boundary_plots
from plotting.violin import violin_plots


def training_algorithm(
    lstm_model,
    svm_model,
    x_loader,
    track_jets_dev_data,
    model_params,
    training_params,
    device,
    print_out="",
    pooling="last",
):
    """Trainging algorithm 1 from paper Tolga: Unsupervised Anomaly Detection With LSTM
    Neural Networks, described in the theory part above


    Args:
        lstm_model (torch.nn.LSTM): LSTM model
        svm_model (sklearn.svm.OneClassSVM): OC-SVM model
        x_loader (list): Dataloader list with batches
        track_jets_dev_data (list): Tracking list for datasequences
        model_params (dict): Parameters used for the models
        training_params (dict): Parameters used for training
        device (torch.device): Cpu or cuda device
        print_out (str, optional): Save output text. Defaults to "".
        pooling (str, optional): Type of pooling used. Defaults to "last".

    Returns:
        Tuple[torch.nn.LSTM, sklearn.svm.OneClassSVM, list, list, bool, str]:
            - LSTM model
            - OC-SVM model
            - list containing cost info of training
            - list containing cost condition info of training
            - Pass or Fail boolean statement
            - Save output text
    """

    ### TRACK TIME ### TODO
    time_track = time.time()

    # Set initial cost
    # TODO set alphas to 1 or 0 so cost_next - cost will be large
    cost = 1e10
    cost_condition_passed_flag = (
        False  # flag to check if cost condition has been satisfied
    )
    min_epochs_patience = training_params["min_epochs"]

    # obtain h_bar from the lstm with theta_0, given the data
    h_bar_list, theta, theta_gradients = calc_lstm_results(
        lstm_model,
        model_params["input_dim"],
        x_loader,
        track_jets_dev_data,
        device,
        pooling=pooling,
    )
    h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

    # list to track cost
    track_cost = []
    track_cost_condition = []

    # loop over k (epochs) for nr. set epochs and unsatisfied cost condition
    k = -1
    while (
        k < min_epochs_patience or cost_condition_passed_flag == False
    ) and k < training_params["max_epochs"]:
        k += 1
        # Copy ai-models to test for next alpha
        svm_model_next = copy(svm_model)
        lstm_model_next = copy(lstm_model)

        # shuffle jets in batches each epoch
        x_loader, track_jets_dev_data = shuffle_batches(x_loader, track_jets_dev_data)

        # keep previous cost result stored
        cost_prev = copy(cost)
        # print("cost before ocsvm\t",cost)

        # obtain alpha_k+1 from the h_bars with SMO through the OC-SVMs .fit()
        svm_model_next.fit(h_bar_list_np)
        alphas = np.abs(svm_model_next.dual_coef_)[0]

        alphas = alphas / np.sum(alphas)  # NOTE: equation 14, sum alphas = 1

        a_idx = svm_model_next.support_

        # print("cost after ocsvm\t", kappa(alphas, a_idx, h_bar_list))

        # obtain theta_k+1 using the optimization algorithm
        lstm_model_next, theta_next = optimization(
            lstm=lstm_model_next,
            alphas=alphas,
            a_idx=a_idx,
            mu=training_params["learning_rate"],
            h_bar_list=h_bar_list,
            theta=theta,
            theta_gradients=theta_gradients,
            device=device,
        )

        # obtain h_bar from the lstm with theta_k+1, given the data
        h_bar_list, theta, theta_gradients = calc_lstm_results(
            lstm_model_next,
            model_params["input_dim"],
            x_loader,
            track_jets_dev_data,
            device=device,
            pooling=pooling,
        )
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

        # print("cost after optim\t", kappa(alphas, a_idx, h_bar_list))

        # obtain the new cost and cost condition given theta_k+1 and alpha_k+1
        cost = kappa(alphas, a_idx, h_bar_list)
        cost_condition = (cost - cost_prev) ** 2

        # track cost and cost_condition
        track_cost.append(cost)
        track_cost_condition.append(cost_condition)

        # check condition algorithm 1, paper Tolga
        if cost_condition < training_params["epsilon"]:
            # check if condition had been satisfied recently
            if cost_condition_passed_flag == False:
                cost_condition_passed_flag = True
                # check if k + patience would be larger than minimum number of epochs
                # Update min_epochs_patience to check if
                if k + training_params["patience"] > training_params["min_epochs"]:
                    min_epochs_patience = k + training_params["patience"]
        else:
            cost_condition_passed_flag = False

        # Check if cost function starts to explode to infinity.
        if np.isnan(track_cost_condition[k]):
            print_out += "\nBroke, for given hyper parameters"
            return (
                lstm_model_next,
                svm_model_next,
                track_cost,
                track_cost_condition,
                False,
                print_out,
            )  # immediately return passed = False

        # use  models if conditions not yet satisfied
        if (
            k < min_epochs_patience or cost_condition_passed_flag == False
        ) and k < training_params["max_epochs"]:
            lstm_model = copy(lstm_model_next)
            svm_model = copy(svm_model_next)

    # add print statements
    ### TRACK TIME ### TODO
    dt = time.time() - time_track
    print_out += f"\nTraining done in: {dt}"

    # check if passed cost condition
    if cost_condition > training_params["epsilon"]:
        print_out += f"\n  Algorithm failed: not done learning in max = {k} epochs"
        passed = False
    else:
        print_out += f"\n  Model done learning in {k} epochs"
        passed = True
    # print_out += f"\n  With cost condition: {abs((cost - cost_prev) / cost_prev)}, vs epsilon: {training_params['epsilon']} "
    print_out += f"\n  With cost condition: {cost_condition}, vs epsilon: {training_params['epsilon']} "

    return lstm_model, svm_model, track_cost, track_cost_condition, passed, print_out


class TRAINING:
    def __init__(self, max_distance) -> None:
        self.max_distance_percentage_anomalies = max_distance

    def run_training(
        self,
        hyper_parameters: dict,
        train_data,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        val_data=None,
        patience=5,
        max_attempts=4,
    ):
        """
        This function searches for the correct hyperparameters.

        It follows the following procedure:
        1. Assign given variables for a run from hyper_parameters.
        2. Prepare data with given parameters.
        3. Try to find distance nu and errors in dev_data prediction < 0.01:
            3.1. Run algorithm 1 from paper tolga..
            Use condition from algorithm to see if the model is still learning.
        4. Save and plot if flags are true
        5. Return distance distance annomalies (With respect to validation if regular training) as loss.

        """
        # track time
        time_track = time.time()

        # variables:
        batch_size = int(hyper_parameters["batch_size"])
        output_dim = int(hyper_parameters["output_dim"])
        layer_dim = int(hyper_parameters["num_layers"])
        dropout = hyper_parameters["dropout"] if layer_dim > 1 else 0
        min_epochs = int(hyper_parameters["min_epochs"])
        learning_rate = hyper_parameters["learning_rate"]
        svm_nu = hyper_parameters["svm_nu"]
        svm_gamma = hyper_parameters["svm_gamma"]
        hidden_dim = int(hyper_parameters["hidden_dim"])
        scaler_id = hyper_parameters["scaler_id"]
        if "variables" in hyper_parameters:
            input_variables = list(hyper_parameters["variables"])
        else:
            input_variables = None
        pooling = hyper_parameters["pooling"]

        # set epsilon and max_epochs
        eps, max_epochs = self.scaled_epsilon_n_max_epochs(learning_rate)

        # output string for printing in terminal:
        print_out = ""

        # Track device
        print_out += "\nDevice: {}".format(device)

        # show used hyper_parameters in terminal
        # sauce https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
        print_out += "\n\nHyper Parameters:\n"
        print_out += "\n".join(
            "  {:10}\t  {}".format(k, v) for k, v in hyper_parameters.items()
        )

        # track time
        time_track = time.time()
        (
            train_data,
            val_data,
            track_jets_train_data,
            track_jets_val_data,
            bf_out_txt,
        ) = self.data_prep_branch_filler(
            train_data, val_data, batch_size, input_variables
        )
        print_out += bf_out_txt

        # track time branch filler
        dt = time.time() - time_track
        print_out += f"\nBranch filler done in: {dt}"

        # scaler: note this has to be saved with the model, to ensure data has the same form.
        if scaler_id == "minmax":
            scaler = MinMaxScaler()
        elif scaler_id == "std":
            scaler = StandardScaler()

        # scale branches
        train_loader, val_loader = self.data_prep_scaling(
            train_data, val_data, scaler, batch_size
        )

        # set model parameters
        input_dim = len(train_data[0])
        model_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout,
            "batch_size": batch_size,
            "device": device,
        }

        # set training parameters
        training_params = {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "epsilon": eps,
            "patience": patience,
        }

        # track time data preparation
        dt = time.time() - time_track
        print_out += f"\nDataprep done in: {dt}"

        # loop for n attempts: if training fails, it attempts more trainings
        n_attempt = 0
        while n_attempt < max_attempts:
            n_attempt += 1

            # declare models
            lstm_model = LSTMModel(**model_params)
            svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="linear")

            # set model to correct device
            lstm_model.to(device)

            # train models
            # try:
            (
                lstm_model,
                svm_model,
                track_cost,
                track_cost_condition,
                passed,
                print_out,
            ) = training_algorithm(
                lstm_model,
                svm_model,
                train_loader,
                track_jets_train_data,
                model_params,
                training_params,
                device,
                print_out,
                pooling,
            )
            # except RuntimeError as e:
            #     passed = False
            #     print("Training algorithm: Cuda error")
            #     logf = open("logfiles/cuda_error.log", "a+")
            #     logf.write(str(e))

            # check if the model passed the training
            loss = 10  # Create standard for saving
            train_success = False

            # check if passed the training
            if passed:
                # Calc loss
                # For regular training also checks if diff_percentage anomalies is small enough.
                loss, train_success = self.calc_loss(
                    train_loader,
                    val_loader,
                    track_jets_train_data,
                    track_jets_val_data,
                    input_dim,
                    lstm_model,
                    svm_model,
                    pooling=pooling,
                    device=device,
                    track_cost=track_cost,
                )

                if train_success:
                    n_attempt = max_attempts

        # track training time and print statement
        dt = time.time() - time_track
        time_str = (
            time.strftime("%H:%M:%S", time.gmtime(dt)) if dt > 60 else f"{dt:.2f} s"
        )
        if train_success:
            print_out += f"\n  With loss: {loss:.4E}"
        print_out += f"\n{'Passed' if train_success else 'Failed'} in: {time_str}"

        # return the model
        lstm_ocsvm = dict({"lstm": lstm_model, "ocsvm": svm_model, "scaler": scaler})

        # save plot data
        cost_data = dict(
            {"cost": track_cost[1:], "cost_condition": track_cost_condition[1:]}
        )

        # print output string
        print(print_out)

        return {
            "loss": loss,
            "final_cost": track_cost[-1],
            "status": STATUS_OK if passed else STATUS_FAIL,
            "model": lstm_ocsvm,
            "hyper_parameters": hyper_parameters,
            "cost_data": cost_data,
            "num_batches": len(train_loader),
        }

    def calc_loss(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        track_jets_val_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
        track_cost,
    ):
        pass

    def data_prep_scaling(self, train_data, val_data, scaler, batch_size):
        pass

    def data_prep_branch_filler(
        self, train_data, val_data, batch_size, input_variables
    ):
        pass

    def scaled_epsilon_n_max_epochs(self, learning_rate):
        """
        Returns an epsilon and max_epochs based on the learning rate.

        Epsilon:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good mach epoch. In addition the learning rate
        determines how much the model changes per time step, and thus determines
        the order of size of the cost_condition, determing when the model has stopped learning.

        In case of Tolga's cost condition: epsilon is chosen as 1/1000 of learning rate
        In case of taking cost condition as a percentage difference of the current cost
        with the previous cost, epsilon is chosen as 10 x learning rate TODO debatable

        Max_epochs:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good max epoch with respect to time management.

        Max_epoch is chosen as the order of size, times a hundred and devided by two, ie
        learning rate = 1e-x, then max epochs = x*100/2=x*50

        Note:
        learning rate must be of the form: 1e-x, where x is a number [0,99]
        """
        order_of_magnitude = int(format(learning_rate, ".1E")[-2:])

        epsilon = 10 ** -(3 + order_of_magnitude)

        more_epochs = 100 * (order_of_magnitude - 3) if order_of_magnitude > 3 else 0
        max_epochs = 700 + more_epochs  # order_of_magnitude * 50

        return epsilon, max_epochs


class HYPER_TRAINING(TRAINING):
    def __init__(self) -> None:
        super().__init__(max_distance=1)

    def calc_loss(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        track_jets_val_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
        track_cost,
    ):
        percentage_anomaly_train = calc_percentage_anomalies(
            train_loader,
            track_jets_train_data,
            input_dim,
            lstm_model,
            svm_model,
            pooling=pooling,
            device=device,
        )

        if percentage_anomaly_train < 0.33:
            passed = True
        else:
            passed = False

        return track_cost[-1], passed

    def data_prep_scaling(self, train_data, val_data, scaler, batch_size):
        train_loader = lstm_data_prep(
            data=train_data, scaler=scaler, batch_size=batch_size, fit_flag=True
        )

        return train_loader, None

    def data_prep_branch_filler(
        self, train_data, val_data, batch_size, input_variables
    ):
        # select only desired input variables
        train_data = train_data[input_variables]

        if type(train_data) is not list:
            train_data = format_ak_to_list(train_data)
        train_data, track_jets_train_data, max_n_batches, _ = branch_filler(
            train_data, batch_size=batch_size, n_features=len(input_variables)
        )
        bf_out_txt = f"\nNr. of train batches: {int(len(train_data) / batch_size)}, out of max.: {max_n_batches}"

        # assign unused data
        track_jets_val_data = None

        return (
            train_data,
            val_data,
            track_jets_train_data,
            track_jets_val_data,
            bf_out_txt,
        )

    def scaled_epsilon_n_max_epochs(self, learning_rate):
        """
        Returns an epsilon and max_epochs based on the learning rate.

        Epsilon:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good mach epoch. In addition the learning rate
        determines how much the model changes per time step, and thus determines
        the order of size of the cost_condition, determing when the model has stopped learning.

        In case of Tolga's cost condition: epsilon is chosen as 1/1000 of learning rate
        In case of taking cost condition as a percentage difference of the current cost
        with the previous cost, epsilon is chosen as 10 x learning rate TODO debatable

        Max_epochs:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good max epoch with respect to time management.

        Max_epoch is chosen as the order of size, times a hundred and devided by two, ie
        learning rate = 1e-x, then max epochs = x*100/2=x*50

        Note:
        learning rate must be of the form: 1e-x, where x is a number [0,99]
        """
        order_of_magnitude = int(format(learning_rate, ".1E")[-2:])

        epsilon = 10 ** -(3 + order_of_magnitude)

        more_epochs = 100 * (order_of_magnitude - 3) if order_of_magnitude > 3 else 0
        max_epochs = 700 + more_epochs  # order_of_magnitude * 50

        return epsilon, max_epochs


class REGULAR_TRAINING(TRAINING):
    def __init__(self) -> None:
        super().__init__(max_distance=0.01)

    def calc_loss(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        track_jets_val_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
        track_cost,
    ):
        percentage_anomaly_validation = calc_percentage_anomalies(
            val_loader,
            track_jets_val_data,
            input_dim,
            lstm_model,
            svm_model,
            pooling=pooling,
            device=device,
        )

        percentage_anomaly_training = calc_percentage_anomalies(
            train_loader,
            track_jets_train_data,
            input_dim,
            lstm_model,
            svm_model,
            device=device,
            pooling=pooling,
        )

        diff_percentage_anomalies = abs(
            percentage_anomaly_training - percentage_anomaly_validation
        )

        # check if distance to svm_nu is smaller than required
        if (
            diff_percentage_anomalies < self.max_distance_percentage_anomalies
            and track_cost[0] != track_cost[-1]
        ):
            succes = True
        else:
            succes = False

        return diff_percentage_anomalies, succes

    def data_prep_scaling(self, train_data, val_data, scaler, batch_size):
        train_loader = lstm_data_prep(
            data=train_data, scaler=scaler, batch_size=batch_size, fit_flag=True
        )

        val_loader = lstm_data_prep(
            data=val_data, scaler=scaler, batch_size=batch_size, fit_flag=False
        )

        return train_loader, val_loader

    def data_prep_branch_filler(
        self, train_data, val_data, batch_size, input_variables
    ):
        if type(train_data) is not list:
            train_data = format_ak_to_list(train_data)
        if type(val_data) is not list:
            val_data = format_ak_to_list(val_data)

        bf_out_txt = ""
        train_data, track_jets_train_data, max_n_train_batches, _ = branch_filler(
            train_data, batch_size=batch_size
        )
        bf_out_txt += f"\nNr. of train batches: {int(len(train_data) / batch_size)}, out of max.: {max_n_train_batches}"
        val_data, track_jets_val_data, max_n_val_batches, _ = branch_filler(
            val_data, batch_size=batch_size
        )
        bf_out_txt += f"\nNr. of validation batches: {int(len(val_data) / batch_size)}, out of max.: {max_n_val_batches}"

        return (
            train_data,
            val_data,
            track_jets_train_data,
            track_jets_val_data,
            bf_out_txt,
        )

    def scaled_epsilon_n_max_epochs(self, learning_rate):
        """
        Returns an epsilon and max_epochs based on the learning rate.

        Epsilon:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good mach epoch. In addition the learning rate
        determines how much the model changes per time step, and thus determines
        the order of size of the cost_condition, determing when the model has stopped learning.

        In case of Tolga's cost condition: epsilon is chosen as 1/1000 of learning rate
        In case of taking cost condition as a percentage difference of the current cost
        with the previous cost, epsilon is chosen as 10 x learning rate TODO debatable

        Max_epochs:
        The learning rate determines how quickly the model learns,
        and thus what we deem a good max epoch with respect to time management.

        Max_epoch is chosen as the order of size, times a hundred and devided by two, ie
        learning rate = 1e-x, then max epochs = x*100/2=x*50

        Note:
        learning rate must be of the form: 1e-x, where x is a number [0,99]
        """
        order_of_magnitude = int(format(learning_rate, ".1E")[-2:])

        epsilon = 10 ** -(3 + order_of_magnitude)

        more_epochs = 10 * (order_of_magnitude - 3) if order_of_magnitude > 3 else 0
        max_epochs = 120 + more_epochs  # order_of_magnitude * 50

        return epsilon, max_epochs


def run_full_training(
    *,
    TRAINING_TYPE: REGULAR_TRAINING or HYPER_TRAINING,
    file_name: str,
    space,
    train_data,
    val_data=None,
    max_evals: int = 4,
    patience: int = 10,
    kt_cut: float = None,  # for dataset, splittings kt > 1.0 GeV
    multicore_flag: bool = False,
    save_results_flag: bool = True,
    plot_flag: bool = True,
    run_notes="",
):
    """
    Run a full training
    TRAINING_TYPE: REGULAR_TRAINING or HYPER_TRAINING: training object
    file_name: file name to train from
    space: set hyper_parameters
    max_evals: max number of evals per trial
    patience: minum number of epochs where cost condition is met before succes condition
    kt_cut: weather to apply a kt_cut
    multicore_flag: for using SparkTrials or Trials, ie multiple cpu cores parallel
    save_results_flag: for saving trials and runtime
    plot_flag: for making cost condition plots, only works if save_results_flag is True
    run_nots: notes on run, added to run_info.p, keep short or leave empty
    """
    # start time
    start_time = time.time()

    # Decide what data to use based on training type
    if TRAINING_TYPE == REGULAR_TRAINING and val_data is None:
        print(
            "ERROR: Regular training requires validation data which has not been provided.\nPlease provide this or switch training type."
        )
        return -1

    # set trials or sparktrials
    if multicore_flag:
        cores = os.cpu_count() if os.cpu_count() < 10 else 10
        trials = SparkTrials(
            parallelism=cores
        )  # run as many trials parallel as the nr of cores available
        print(f"Running {max_evals} evaluations, on {cores} cores:\n")
    else:
        trials = Trials()  # NOTE keep for debugging since can't do with spark trials

    # Create training object
    training = TRAINING_TYPE()

    # use correct device:
    if multicore_flag:
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # hyper tuning and evaluation
    best = fmin(
        partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
            training.run_training,
            train_data=train_data,
            val_data=val_data,
            patience=patience,
            device=device,
        ),
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    # saving spark_trials as dictionaries
    # source https://stackoverflow.com/questions/63599879/can-we-save-the-result-of-the-hyperopt-trials-with-sparktrials
    pickling_trials = dict()
    for k, v in trials.__dict__.items():
        if not k in ["_spark_context", "_spark"]:
            pickling_trials[k] = v

    # collect df and print best model(s)
    df, min_val, min_df, parameters = trials_df_and_minimum(pickling_trials, "loss")

    # print run statement
    print(
        f"\n{'Hypertuning' if TRAINING_TYPE == HYPER_TRAINING else 'Regular training'} completed on dataset:\n\t{file_name}"
    )

    # check to save results
    if save_results_flag:
        # set out file to job_id for parallel computing
        job_id = os.getenv("PBS_JOBID")
        if job_id:
            job_id = job_id.split(".")[0]
        else:
            job_id = time.strftime("%y_%m_%d_%H%M")

        out_file = f"storing_results/trials_test_{job_id}.p"

        # save trials as pickling_trials object
        torch.save(pickling_trials, open(out_file, "wb"))
        print(f"Stored results in:\n\t{out_file}")

        # check to make plots
        if plot_flag:
            cost_condition_plots(pickling_trials, job_id)
            violin_plots(
                df, min_val, min_df, parameters, [job_id], "loss"
            ) if TRAINING_TYPE == HYPER_TRAINING else None
            # svm_boundary_plots(pickling_trials, job_id, train_data)
            print(
                f"Plotting complete, stored results at:\n\toutput/cost_condition_{job_id}/\n\toutput/violin_plots_{job_id}/"
            )

        # store run info
        run_time = time.time() - start_time
        run_info = f"{job_id}\ton: {file_name}\truntime: {run_time:.2f} s"
        run_info = (
            run_info + f"\tnotes: {run_notes}\n" if run_notes else run_info + "\n"
        )
        with open("storing_results/run_info.p", "a+") as f:
            f.write(run_info)
        print(f"\nCompleted run in: {run_time:.2f} seconds\n\ton job: {job_id}")
