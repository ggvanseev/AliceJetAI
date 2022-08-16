import torch


def get_mean_theta_gradients(theta_gradients, n_datasets):
    for key1, value1 in theta_gradients.items():
        for key2, value2 in value1.items():
            theta_gradients[key1][key2] = theta_gradients[key1][key2] / n_datasets

    return theta_gradients


def calc_lstm_results(
    lstm_model,
    input_dim,
    data_loader,
    track_jets_data,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    pooling="last",
):
    """Obtain h_bar states from the lstm with the data

    Args:
        lstm_model (LSTMModel): the LSTM model
        train_loader (torch.utils.data.dataloader.DataLoader): object by PyTorch, stores
                the data
        model_params ():
        track_jets_train_data ():
        batch_size ():
        device ():

    Returns:
        torch.Tensor: contains the h_bar results from the LSTM
        dict: new theta containing weights and biases
    """
    h_bar_list = list()

    # make sure theta is only generetad once.
    theta = None
    theta_gradients = None
    
    # set correct device to LSTM model
    if device != lstm_model.set_device:
        print(f"LSTM model device set to: {device}")
        lstm_model.set_device = device

    i = 0
    for x_batch, y_batch in data_loader:
        jet_track_local = track_jets_data[i]
        i += 1

        # x_batch as input for lstm, pytorch says shape = [sequence_length, batch_size, n_features]
        # x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(device)
        x_batch = x_batch.view([len(x_batch), -1, input_dim]).to(device)
        y_batch = y_batch.to(device)
        
        ### Train step
        # set model to train
        # lstm_model.train()  # TODO should this be off so the backward() call in the forward pass does not update the weights?

        # Makes predictions
        h_bar, theta, theta_gradients = lstm_model(
            x_batch, jet_track_local, pooling=pooling, theta=theta, theta_gradients=theta_gradients
        )

        # h_bar_list.append(h_bar) # TODO, h_bar is not of fixed length! solution now: append all to list, then vstack the list to get 2 axis structure
        h_bar_list.append(h_bar)

    # Get mean of theta gradients
    theta_gradients = get_mean_theta_gradients(
        theta_gradients, n_datasets=len(data_loader)
    )

    return (
        torch.vstack([h_bar[-1] for h_bar in h_bar_list]),
        theta,
        theta_gradients,
    )  # take h_bar[-1], to take the last layer as output
