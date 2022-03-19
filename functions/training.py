from msilib.schema import Error
import torch
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# from sklearn.externals import joblib
# import joblib

import time

from functions.data_manipulation import (
    branch_filler,
    lstm_data_prep,
    h_bar_list_to_numpy,
    scaled_epsilon_n_max_epochs,
    format_ak_to_list,
    train_dev_test_split,
    trials_df_and_minimum,
)

from functions.optimization_orthogonality_constraints import (
    kappa,
    optimization,
)

from functions.run_lstm import calc_lstm_results

from plotting.general import plot_cost_vs_cost_condition

from functions.validation import calc_percentage_anomalies
from ai.model_lstm import LSTMModel

# from autograd import elementwise_grad as egrad

from copy import copy

from functions.data_loader import load_n_filter_data
from plotting.general import cost_condition_plots, violin_plots

# from autograd import elementwise_grad as egrad


from hyperopt import (
    fmin,
    tpe,
    STATUS_OK,
    STATUS_FAIL,
    Trials,
    SparkTrials,
)  # Cite: Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
from functools import partial

import os


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
    """
    Trainging algorithm 1 from paper Tolga: Unsupervised Anomaly Detection With LSTM Neural Networks
    """
    # path for model - only used for saving
    # model_path = f'models/{lstm_model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    ### ALGORITHM START ###

    ### TRACK TIME ### TODO
    time_at_step = time.time()

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

    ### TRACK TIME ### TODO
    # dt = time.time() - time_at_step
    # time_at_step = time.time()
    # print(f"Obtained first h_bars, done in: {dt}")

    # loop over k (epochs) for nr. set epochs and unsatisfied cost condition
    k = -1
    while (
        k < min_epochs_patience or cost_condition_passed_flag == False
    ) and k < training_params["max_epochs"]:
        k += 1

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Start of loop, done in: {dt} \t epoch {k}")

        # keep previous cost result stored
        cost_prev = copy(cost)

        # obtain alpha_k+1 from the h_bars with SMO through the OC-SVMs .fit()
        svm_model.fit(h_bar_list_np)
        alphas = np.abs(svm_model.dual_coef_)[0]

        alphas = alphas / np.sum(alphas)  # NOTE: equation 14, sum alphas = 1

        a_idx = svm_model.support_

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained alphas, done in: {dt}")

        # obtain theta_k+1 using the optimization algorithm
        lstm_model, theta_next = optimization(
            lstm=lstm_model,
            alphas=alphas,
            a_idx=a_idx,
            mu=training_params["learning_rate"],
            h_bar_list=h_bar_list,
            theta=theta,
            theta_gradients=theta_gradients,
            device=device,
        )

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained thetas, done in: {dt}")

        # obtain h_bar from the lstm with theta_k+1, given the data
        h_bar_list, theta, theta_gradients = calc_lstm_results(
            lstm_model,
            model_params["input_dim"],
            x_loader,
            track_jets_dev_data,
            device=device,
            pooling=pooling,
        )
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, device)

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained h_bar, done in: {dt}")

        # obtain the new cost and cost condition given theta_k+1 and alpha_k+1
        cost = kappa(alphas, a_idx, h_bar_list)
        cost_condition = (cost - cost_prev) ** 2

        ### TRACK TIME ### TODO
        # dt = time.time() - time_at_step
        # time_at_step = time.time()
        # print(f"Obtained cost, done in: {dt}")

        # track cost and cost_condition
        track_cost.append(cost)
        track_cost_condition.append(cost_condition)

        # check condition algorithm 1, paper Tolga
        if abs((cost - cost_prev) / cost_prev) < training_params["epsilon"]:
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
                lstm_model,
                svm_model,
                track_cost,
                track_cost_condition,
                False,
                print_out,
            )  # immediately return passed = False

    print_out += f"\nfrac diff: {(cost - cost_prev) / cost_prev},  eps: {training_params['epsilon']} "
    if abs((cost - cost_prev) / cost_prev) > training_params["epsilon"]:
        print_out += "\nAlgorithm failed: not done learning in max epochs."
        passed = False
    else:
        print_out += f"\nModel done learning in {k} epochs."
        passed = True

    return lstm_model, svm_model, track_cost, track_cost_condition, passed, print_out


class TRAINING:
    def __init__(self, max_distance) -> None:
        self.max_distance_percentage_anomalies = max_distance

    def run_training(
        self,
        hyper_parameters: dict,
        train_data,
        val_data=None,
        plot_flag: bool = False,
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
        # Track time
        time_track = time.time()

        # Variables:
        batch_size = int(hyper_parameters["batch_size"])
        output_dim = int(hyper_parameters["output_dim"])
        layer_dim = int(hyper_parameters["num_layers"])
        dropout = hyper_parameters["dropout"] if layer_dim > 1 else 0  # TODO
        min_epochs = int(hyper_parameters["min_epochs"])
        learning_rate = hyper_parameters["learning_rate"]
        svm_nu = hyper_parameters["svm_nu"]
        svm_gamma = hyper_parameters["svm_gamma"]
        hidden_dim = int(hyper_parameters["hidden_dim"])
        scaler_id = hyper_parameters["scaler_id"]
        input_variables = list(hyper_parameters["variables"])
        pooling = hyper_parameters["pooling"]

        # Set epsilon and max_epochs
        eps, max_epochs = scaled_epsilon_n_max_epochs(learning_rate)

        # output string for printing in terminal:
        print_out = ""

        # Show used hyper_parameters in terminal
        # sauce https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
        print_out += "\n\nHyper Parameters:\n"
        print_out += "\n".join(
            "  {:10}\t  {}".format(k, v) for k, v in hyper_parameters.items()
        )

        # use correct device:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print_out += "\nDevice: {}".format(device)

        # prepare data for usage
        # dev_data_copy = copy(dev_data)  # save this to check the error of data[] TODO
        # try:
        #     dev_data, track_jets_dev_data = branch_filler(dev_data, batch_size=batch_size)
        # except TypeError:
        #     print("Could not create jet branch with given data and parameters!")
        #     return (
        #         10  # for "loss", since this will be added to the 1st column of the result
        #     )

        time_track = time.time()

        try:
            train_data, val_data, track_jets_train_data = self.data_prep_branch_filler(
                train_data, val_data, batch_size, input_variables
            )
        except:
            print("Branch filler failed")
            return {
                "loss": 10,
                "final_cost": 10,
                "status": STATUS_FAIL,
                "model": 10,
                "hyper_parameters": hyper_parameters,
                "cost_data": 10,
                "num_batches": batch_size,
            }

        dt = time.time() - time_track
        print(f"Branch filler, done in: {dt}")

        # Note this has to be saved with the model, to ensure data has the same form.
        if scaler_id == "minmax":
            scaler = MinMaxScaler()
        elif scaler_id == "std":
            scaler = StandardScaler()

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

        ### TRACK TIME ### TODO
        dt = time.time() - time_track
        print_out += f"\nDataprep, done in: {dt}"

        n_attempt = 0
        while n_attempt < max_attempts:
            n_attempt += 1

            # Declare models
            lstm_model = LSTMModel(**model_params)
            svm_model = OneClassSVM(nu=svm_nu, gamma=svm_gamma, kernel="linear")

            # set model to correct device
            lstm_model.to(device)

            try:
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
            except RuntimeError as e:
                passed = False
                logf = open("logfiles/cuda_error.log", "w")
                logf.write(str(e))

            # check if the model passed the training
            diff_percentage_anomalies = 10  # Create standard for saving
            train_success = False

            if passed:
                diff_percentage_anomalies = self.calc_diff_percentage(
                    train_loader,
                    val_loader,
                    track_jets_train_data,
                    input_dim,
                    lstm_model,
                    svm_model,
                    pooling=pooling,
                    device=device,
                )

                # Check if distance to svm_nu is smaller than required
                if (
                    diff_percentage_anomalies < self.max_distance_percentage_anomalies
                    and track_cost[0] != track_cost[-1]
                ):
                    n_attempt = max_attempts
                    train_success = True

        # training time and print statement
        dt = time.time() - time_track
        time_str = (
            time.strftime("%H:%M:%S", time.gmtime(dt)) if dt > 60 else f"{dt:.2f} s"
        )
        print_out += f"\n{'Passed' if train_success else 'Failed'} in: {time_str}"
        if train_success:
            print_out += f"\twith loss: {diff_percentage_anomalies:.4E}"

        if plot_flag:
            # plot cost condition and cost function
            title_plot = f"plot_with_{max_epochs}_epochs_{batch_size}_batch_size_{learning_rate}_learning_rate_{svm_gamma}_svm_gamma_{svm_nu}_svm_nu_{diff_percentage_anomalies}_distance_nu"
            plot_cost_vs_cost_condition(
                track_cost=track_cost,
                track_cost_condition=track_cost_condition,
                title_plot=title_plot,
                save_flag=True,
            )

        # return the model
        lstm_ocsvm = dict({"lstm": lstm_model, "ocsvm": svm_model, "scaler": scaler})

        # save plot data
        cost_data = dict(
            {"cost": track_cost[1:], "cost_condition": track_cost_condition[1:]}
        )

        # print output string
        print(print_out)

        return {
            "loss": diff_percentage_anomalies,
            "final_cost": track_cost[-1],
            "status": STATUS_OK if passed else STATUS_FAIL,
            "model": lstm_ocsvm,
            "hyper_parameters": hyper_parameters,
            "cost_data": cost_data,
            "num_batches": len(train_loader),
        }

    def calc_diff_percentage(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
    ):
        pass

    def data_prep_scaling(self, train_data, val_data, scaler, batch_size):
        pass

    def data_prep_branch_filler(
        self, train_data, val_data, batch_size, input_variables
    ):
        pass


class HYPER_TRAINING(TRAINING):
    def __init__(self) -> None:
        super().__init__(max_distance=1000)

    def calc_diff_percentage(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
    ):
        return calc_percentage_anomalies(
            train_loader,
            track_jets_train_data,
            input_dim,
            lstm_model,
            svm_model,
            device=device,
            pooling=pooling,
        )

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

        train_data = format_ak_to_list(train_data)
        train_data, track_jets_train_data, max_n_batches, _ = branch_filler(
            train_data, batch_size=batch_size, n_features=len(input_variables)
        )
        print(f"\nMax number of batches: {max_n_batches}")

        return train_data, val_data, track_jets_train_data


class REGULAR_TRAINING(TRAINING):
    def __init__(self) -> None:
        super().__init__(max_distance=0.01)

    def calc_diff_percentage(
        self,
        train_loader,
        val_loader,
        track_jets_train_data,
        input_dim,
        lstm_model,
        svm_model,
        pooling,
        device,
    ):
        percentage_anomaly_validation = calc_percentage_anomalies(
            val_loader,
            track_jets_train_data,
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

        return diff_percentage_anomalies

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
        train_data = format_ak_to_list(train_data)
        val_data = format_ak_to_list(val_data)

        train_data, track_jets_train_data, max_n_train_batches, _ = branch_filler(
            train_data, batch_size=batch_size
        )
        print(f"\nMax number of batches: {max_n_train_batches}")
        val_data, track_jets_val_data, max_n_val_batches, _ = branch_filler(
            val_data, batch_size=batch_size
        )
        print(f"\nMax number of batches: {max_n_val_batches}")

        return train_data, val_data, track_jets_train_data


def run_full_training(
    *,
    TRAINING_TYPE: REGULAR_TRAINING or HYPER_TRAINING,
    file_name: str,
    space,
    max_evals: int = 4,
    patience: int = 5,
    kt_cut=False,  # for dataset, splittings kt > 1.0 GeV
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

    # Load and filter data for criteria eta and jetpt_cap
    g_recur_jets, _ = load_n_filter_data(file_name, kt_cut=kt_cut)
    print("Loading data complete")
    # split data
    split_train_data, split_dev_data, split_val_data = train_dev_test_split(
        g_recur_jets, split=[0.8, 0.1]
    )
    print("Splitting data complete")

    # Decide what data to use based on training type
    if TRAINING_TYPE == REGULAR_TRAINING:
        train_data = split_train_data
        val_data = split_val_data
    else:
        train_data = split_dev_data
        val_data = None

    # set trials or sparktrials
    if multicore_flag:
        cores = os.cpu_count() if os.cpu_count() < 10 else 10
        trials = SparkTrials(
            parallelism=cores
        )  # run as many trials parallel as the nr of cores available
        print(f"Hypertuning {max_evals} evaluations, on {cores} cores:\n")
    else:
        trials = Trials()  # NOTE keep for debugging since can't do with spark trials

    # Create training object
    training = TRAINING_TYPE()

    # hyper tuning and evaluation
    best = fmin(
        partial(  # Use partial, to assign only part of the variables, and leave only the desired (args, unassiged)
            training.run_training,
            train_data=train_data,
            val_data=val_data,
            plot_flag=plot_flag,
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
            job_id = job_id.split(".")[0]
        else:
            job_id = time.strftime("%d_%m_%y_%H%M")

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
        run_info = (
            run_info + f"\tnotes: {run_notes}\n" if run_notes else run_info + "\n"
        )
        with open("storing_results/run_info.p", "a+") as f:
            f.write(run_info)
        print(f"\nCompleted run in: {run_time}")

        # load torch.load(r"storing_results\trials_test.p",map_location=torch.device('cpu'), pickle_module=pickle)
