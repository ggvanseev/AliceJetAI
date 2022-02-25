import torch
import numpy as np

from functions.data_manipulation import (
    lstm_data_prep,
    branch_filler,
    h_bar_list_to_numpy,
)


class LSTM_OCSVM_CLASSIFIER:
    def __init__(
        self,
        oc_svm,
        lstm,
        batch_size,
        scaler,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        self.oc_svm = oc_svm
        self.lstm = lstm.to(device)
        self.batch_size = batch_size
        self.device = device
        self.scaler = scaler

    def anomaly_classifaction(self, data):
        """
        Classifies anomalies
        data: data set to be classified
        returns: classifaction, percentage anomaly

        """

        data, track_jets_data, _ = branch_filler(data, batch_size=self.batch_size)

        data_loader = lstm_data_prep(
            data=data, scaler=self.scaler, batch_size=self.batch_size
        )

        input_dim = len(data[0])

        h_bar_list = []
        with torch.no_grad():
            i = 0
            for x_batch, y_batch in data_loader:
                jet_track_local = track_jets_data[i]
                i += 1

                x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(self.device)
                y_batch = y_batch.to(self.device)

                # Makes predictions, and don't use backpropagation
                hn = self.lstm(x_batch, backpropagation_flag=False)

                # get mean pooled hidden states
                h_bar = hn[:, jet_track_local]

                h_bar_list.append(h_bar)

        # Take last layer
        h_bar_list = torch.vstack([h_bar[-1] for h_bar in h_bar_list])
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list, self.device)

        # get prediction
        classifaction = self.oc_svm.predict(h_bar_list_np)

        # count anomalies
        n_anomaly = np.count_nonzero(classifaction == -1)

        fraction_anomaly = n_anomaly / len(classifaction)

        return classifaction, fraction_anomaly
