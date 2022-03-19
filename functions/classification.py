import awkward as ak
import torch
import numpy as np

from functions.data_manipulation import (
    lstm_data_prep,
    branch_filler,
    h_bar_list_to_numpy,
    format_ak_to_list,
)

from functions.run_lstm import calc_lstm_results

import pickle


class LSTM_OCSVM_CLASSIFIER:
    def __init__(
        self,
        oc_svm,
        lstm,
        batch_size,
        scaler,
        pooling="last",
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        self.oc_svm = oc_svm
        self.lstm = lstm.to(device)
        self.batch_size = batch_size
        self.device = device
        self.scaler = scaler
        self.pooling = pooling

    def anomaly_classifaction(
        self, data: ak, zeros_test_flag=False, nines_test_flag=False
    ):
        """
        Classifies anomalies, with the possibility of setting a test
        data: data set to be classified
        returns: classifaction, percentage anomaly

        """
        if zeros_test_flag:
            data = (np.zeros([500, int(self.batch_size / 100), len(data)])).tolist()
        elif nines_test_flag:
            data = (np.zeros([500, int(self.batch_size / 100), len(data)]) + 9).tolist()
        else:
            if type(data) != list:
                data = format_ak_to_list(data)

        try:
            data_in_branches, track_jets_data, _, jets_index = branch_filler(
                data, batch_size=self.batch_size
            )
        except TypeError:
            print(
                "LSTM_OCSVM_CLASSIFIER: TypeError 'cannot unpack non-iterable int object'\nBranch filler failed"
            )
            classification = -1
            fraction_anomaly = -1
            return classification, fraction_anomaly

        data_loader = lstm_data_prep(
            data=data_in_branches,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )

        input_dim = len(data_in_branches[0])

        h_bar_list, _, _ = calc_lstm_results(
            self.lstm,
            input_dim,
            data_loader,
            track_jets_data,
            self.device,
            self.pooling,
        )

        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, self.device)

        # get prediction
        classifaction = self.oc_svm.predict(h_bar_list_np)

        # count anomalies
        n_anomaly = np.count_nonzero(classifaction == -1)

        fraction_anomaly = n_anomaly / len(classifaction)

        return classifaction, fraction_anomaly, jets_index


class CLASSIFICATION_CHECK:
    def __init__(self) -> None:
        pass

    def classifaction_test(self, trials, zeros_flag, nines_test_flag):
        anomaly_tracker = np.zeros(len(trials))
        for i in range(len(trials)):
            # select model
            model = trials[i]["result"]["model"]

            # TODO check for bad models
            if model == 10:
                continue

            lstm_model = model["lstm"]  # note in some old files it is lstm:
            ocsvm_model = model["ocsvm"]
            scaler = model["scaler"]

            # get hyper parameters
            batch_size = int(trials[i]["result"]["hyper_parameters"]["batch_size"])
            input_variables = list(trials[i]["result"]["hyper_parameters"]["variables"])

            classifier = LSTM_OCSVM_CLASSIFIER(
                oc_svm=ocsvm_model,
                lstm=lstm_model,
                batch_size=batch_size,
                scaler=scaler,
            )

            _, anomaly_tracker[i], _ = classifier.anomaly_classifaction(
                data=input_variables,
                zeros_test_flag=zeros_flag,
                nines_test_flag=nines_test_flag,
            )

            if anomaly_tracker[i] == 0:
                anomaly_tracker[i] = "nan"

        return np.argwhere(np.isnan(anomaly_tracker)).T[0]

    def classifaction_all_nines_test(self, trials):
        """
        Dummy jets with value nine, to exclude ai-models than don't look at the content of the jets.
        """
        return self.classifaction_test(trials, zeros_flag=False, nines_test_flag=True)

    def classifaction_all_zeros_test(self, trials):
        return self.classifaction_test(trials, zeros_flag=True, nines_test_flag=False)


def get_anomalies(jets, job_id, trials, file_name, jet_info=""):
    # Create storing containers
    anomaly_tracker = np.zeros(len(trials))
    classifaction_tracker = dict()
    jets_index_tracker = dict()

    # Run through all trials
    for i in range(len(trials)):
        # select model
        model = trials[i]["result"]["model"]

        lstm_model = model["lstm"]  # note in some old files it is lstm:
        ocsvm_model = model["ocsvm"]
        scaler = model["scaler"]

        # get hyper parameters
        batch_size = int(trials[i]["result"]["hyper_parameters"]["batch_size"])
        input_variables = list(trials[i]["result"]["hyper_parameters"]["variables"])

        classifier = LSTM_OCSVM_CLASSIFIER(
            oc_svm=ocsvm_model, lstm=lstm_model, batch_size=batch_size, scaler=scaler
        )

        (
            classifaction_tracker[i],
            anomaly_tracker[i],
            jets_index_tracker[i],
        ) = classifier.anomaly_classifaction(data=jets[input_variables])

        print(
            f"Percentage classified as anomaly: {np.round(anomaly_tracker[i]*100,2) }%, where the model has a nu of {trials[i]['result']['hyper_parameters']['svm_nu']}"
        )

    print(
        f"Average percentage anomalys: {np.round(np.nanmean(anomaly_tracker)*100,2)} +\- {np.round(np.nanstd(anomaly_tracker)*100,2)}%"
    )

    # store results
    storing = {
        "jets_index": jets_index_tracker,
        "percentage_anomalies": anomaly_tracker,
        "classifaction_annomaly": classifaction_tracker,
        "data": jets,
        "file": file_name,
    }
    pickle.dump(
        storing,
        open(f"storing_results/anomaly_classification_{jet_info}_{job_id}.pkl", "wb"),
    )
