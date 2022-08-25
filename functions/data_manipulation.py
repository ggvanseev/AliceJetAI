import torch
import awkward as ak
from copy import copy
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from itertools import compress

import branch_names as na

# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings

# ignore numba warnings in terms of depraciation
# warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
# warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


def format_ak_to_list(arr: ak.Array) -> list:
    """Function to reformat Awkward arrays g_jets_recur and q_jets_recur of
    load_n_filter_data, which is required for a few purposes:
        - Awkward arrays (assumably) not accepted as input for LSTM.
        - Removal of empty event entries of the dataset.
        - Reshape to: nr. variables x nr. splitings
    Args:
        arr (ak.Array): Input Awkward array containing recursive jet data
    Returns:
        list: Output list suitable as LSTM input, shape [dr[...], pt[...], z[...]]
    """

    # awkward.to_list() creates dictionaries, reform to list only
    lst = [list(x.values()) for x in ak.to_list(arr)]
    # remove weird nestedness, e.g. dr[[...]]
    # lst = [[y[0] for y in x] for x in lst if x and any(x) and any(x[0])]
    # remove empty lists:
    # lst = [x for x in lst if len(x[0]) > 0]
    # transpose remainder to get correct shape
    lst = [list(map(list, zip(*x))) for x in lst]
    return lst


def train_dev_test_split(dataset, split=[0.8, 0.1]):
    """
    Split is the percentage cut of for selecting training data.
    Thus split 0.8 means 80% of data is considered for training.
    Second split of 0.1 means 10% of data is considered for development.
    """
    max_train_index = int(len(dataset) * split[0])
    max_dev_index = int(len(dataset) * (split[0] + split[1]))
    train_data = dataset[0:max_train_index]
    dev_data = dataset[max_train_index:max_dev_index]
    test_data = dataset[max_dev_index:]

    return train_data, dev_data, test_data


def copy_dataset(dataset):
    """
    Function copies the dataset such that it has similar qualities to a deepcopy,
    e.g. manipulation of the copy won't affect the original
    """
    temp_dataset = dict()
    temp_dataset2 = dict()

    temp_dataset[0] = copy(dataset[0])
    temp_dataset[1] = copy(dataset[1])

    temp_dataset2[0] = copy(dataset[0])
    temp_dataset2[1] = copy(dataset[1])

    return temp_dataset, temp_dataset2


def branch_filler(original_dataset, batch_size, n_features=3, max_trials=100):
    """
    Tries to fill data into batches, drop left over data.
    Also trackts where the jets are in the branch.

    Returns dataset that fits into branches, and a list to track where the jets are in the dataset.

    Don't use ordering or sorting to avoid introducing biases into the lstm, the sample has a chaotic
    """
    # make safety copy to avoid changing results
    original_data = copy(original_dataset)

    # count all values (, is indicative of a value), and divide by n_features to prevent double counting splits
    max_n_batches = int(str(original_data).count(",") / n_features / batch_size)

    # batches, is a list with all the batches
    batches = []
    # Track_jets_in_batch tracks where the last split of the jet is located in the branch
    track_jets_in_batch = []

    # Track index
    track_index = list()

    original_index = list(range(0, len(original_data)))

    dataset = [original_data, original_index]

    i = -1
    while i < max_n_batches:
        i += 1

        # space count tracks if branch is filled to max
        space_count = batch_size

        # make copies of the dataset to be able to remove elemnts while trying to fill branches
        # without destroyting original dataset
        temp_dataset, temp_dataset2 = copy_dataset(dataset)

        # local trakcers of batches ad jets_in_batch
        batch = list()
        jets_in_batch = list()
        index_in_batch = list()

        # variables that track progress
        add_branch_flag = True
        trials = 0

        # Check if batch is filled
        while space_count > 0:

            # loop over available jets
            j = -1
            len_temp_dataset = len(temp_dataset[0])
            while j < len_temp_dataset:
                j += 1

                # check if temp_dataset2 still has elements
                if len(temp_dataset2[0]) < 1:
                    add_branch_flag = False
                    i = max_n_batches
                    space_count = 0
                    break

                # use a normal jet finding sequence when there is a lot of space
                if space_count > 12:
                    jet = temp_dataset[0][j]
                # when approaching end of the list, can be a long time, thus use arrays to find
                # next jet more easily
                elif space_count >= min(map(len, temp_dataset2[0])):
                    jet_index = np.argmax(
                        np.array(list(map(len, temp_dataset2[0])), dtype=object)
                        <= space_count
                    )
                    jet = temp_dataset2[0][jet_index]
                else:
                    jet = temp_dataset[0][j]

                # Add jet to batch if possible
                if space_count >= len(jet):
                    batch.append(jet)

                    # index position of jet in dataset
                    index_pos = temp_dataset2[0].index(jet)
                    # Append original index position for later tracking
                    index_in_batch.append(temp_dataset2[1][index_pos])

                    try:
                        # Delete element by index position
                        del temp_dataset2[0][index_pos]
                        del temp_dataset2[1][index_pos]
                    except ValueError:
                        print(
                            "branch_filler: ValueError 'list.remove(x): x not in list'\nattempted to remove a jet that was no longer in temp_dataset2"
                        )
                    space_count -= len(jet)
                    jets_in_batch.append(
                        batch_size - space_count - 1
                    )  # Position of the last split of jet

                elif space_count == 0:
                    break

                # Check if anything could be added at all to avoid unnecesary loop
                # (if not this means looping doesn't add anything, because no free spots can be filled up any more
                elif space_count < len(min(temp_dataset2[0])) and space_count > 0:
                    # remove first entry from dataset to see try with different initial condition for filling list
                    del dataset[0][0]
                    del dataset[1][0]

                    # Reset to original values
                    # Space count tracks if branch is filled to max
                    space_count = batch_size

                    # make copies of the dataset to be able to remove elemnts while trying to fill branches
                    # without destroyting original dataset
                    temp_dataset, temp_dataset2 = copy_dataset(dataset)

                    # local trakcers of batches ad jets_in_batch
                    batch = list()
                    jets_in_batch = list()
                    index_in_batch = list()

                    # update trials:
                    trials += 1

                    if (
                        trials == max_trials
                        or int(str(dataset[0]).count(",") / n_features) < batch_size
                    ):
                        add_branch_flag = False
                        i = max_n_batches
                        space_count = 0

                    break

                # If at the end of the temp_dataset update it to try to fill in free spots
                elif j == len_temp_dataset - 1:
                    temp_dataset[0] = copy(temp_dataset2[0])
                    temp_dataset[1] = copy(temp_dataset2[1])
                    break

        if add_branch_flag:
            batches.append([y for x in batch for y in x])  # remove list nesting
            track_jets_in_batch.append(
                list(dict.fromkeys(jets_in_batch))
            )  # Only unique values (empty jet can add "end" jet)
            dataset = temp_dataset2
            track_index.append(index_in_batch)

    # Remove list nesting of branches (will be restored by DataLoader from torch if same
    # batch size and shuffle=False is used)
    batches = [y for x in batches for y in x]
    track_index = [y for x in track_index for y in x]

    # check if not successful
    if not batches:
        return -1

    return batches, track_jets_in_batch, max_n_batches, track_index


def single_branch(data):
    """Create one single branch from all data, uses are for classification.
    Reason for a single branch is that branch filler removes jets from 
    original dataset in order to successfully fill branches of a specific size.
    During testing, especially, losing numerous jets is not desirable."""
    
    # create new tracks
    current_pos = -1
    track_jets_in_batch = list()
    for length in [len(x) for x in data]:
        current_pos += length
        track_jets_in_batch.append(current_pos)
    
    track_index = [x for x in range(len(data))] # simple list of 1 -> total
    track_jets_in_batch = [track_jets_in_batch] # put in list for "batches"
    data = [x for y in data for x in y] # flatten = single batch
    batch_size = len(data) # size of the single batch
    
    return data, batch_size, track_jets_in_batch, track_index
    
    
def shuffle_batches(batches, track_jets_in_batch, device, shuffle=False):
    """Function that shuffles batches according to batch structure
    built by the branch_filler

    Args:
        batches (_type_): _description_
        track_jets_in_batch (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
        
    batches_shuffled = list() # contains newly shuffled batches
    track_jets_shuffled = list() # ontains newly shuffled tracks

    for (batch, _), tracks in zip(batches, track_jets_in_batch):
        # convert to cpu
        if device.type != "cpu":
            batch = batch.to(torch.device("cpu"))
        
        # rebuild batches in order to shuffle them
        batch_rebuilt = [
            batch[tracks[i - 1] + 1 if i > 0 else None : tracks[i] + 1]
            for i in range(len(tracks))
        ]

        # shuffle batches
        random.shuffle(batch_rebuilt)

        # create new tracks
        current_pos = -1
        new_tracks = list()
        for length in [len(x) for x in batch_rebuilt]:
            current_pos += length
            new_tracks.append(current_pos)

        # add to lists
        track_jets_shuffled.append(new_tracks)
        batches_shuffled.extend([x for y in batch_rebuilt for x in y])

    # convert back to torch tensor
    data = torch.stack(batches_shuffled)

    # Data loader needs labels, but isn't needed for unsupervised, thus fill in data for labels to run since it won't be used.
    data = TensorDataset(data, data)
    data = DataLoader(data, batch_size=len(batch), shuffle=shuffle)

    return data, track_jets_shuffled # TODO indices as well?


def lstm_data_prep(*, data, scaler, batch_size, fit_flag=False, shuffle=False):
    """
    Returns a DataLoader class to work with the large datasets in skilearn LSTM
    """

    # Check if it is the first data for the scalar, if so fit scalar to this data.
    if fit_flag:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    # Make data in tensor format
    data = torch.Tensor(data)

    # Data loader needs labels, but isn't needed for unsupervised, thus fill in data for labels to run since it won't be used.
    data = TensorDataset(data, data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def get_weights(model, hidden_dim):
    """
    Returns the weight ordered as in the paper(see Tolga)
    Using the scheme below and the knowledge that the weights in the paper (see Tolga, anomaly) correspond as the following:
    W(x)=Wix, R(x)=Whx and b(x) = bix + bhx, where x is of {I,f,z/g,o}. Where z=g respectively.


    LSTM.weight_ih_l[k] – the learnable input-hidden weights of the \text{k}^{th}k th
    layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is (4*hidden_size, num_directions * hidden_size).
    If proj_size > 0 was specified, the shape will be (4*hidden_size, num_directions * proj_size) for k > 0

    ~LSTM.weight_hh_l[k] – the learnable hidden-hidden weights of the \text{k}^{th}k th
     layer (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size). If proj_size > 0 was specified, the shape will be (4*hidden_size, proj_size).

    ~LSTM.bias_ih_l[k] – the learnable input-hidden bias of the \text{k}^{th}kthlayer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)

    ~LSTM.bias_hh_l[k] – the learnable hidden-hidden bias of the \text{k}^{th}kth
    layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)
    source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    """
    # Lists that make it easy to select the matching weight matrixes that are stored in one tensor/matrix by pytorch model
    weight_type_list = ["i", "f", "g", "o"]
    weight_type_selector = [k * hidden_dim for k in [0, 1, 2, 3]]
    weight_type_selector.append(None)

    # Coresponds to W,R and B respectively:
    w = dict()
    r = dict()
    bi = dict()
    bh = dict()

    # Loop over the total present layers
    for i in range(model.num_layers):
        # loop over all weight types
        for j in range(4):
            # Make sure all names have the same string length between beginning and first {}
            w[f"w__{weight_type_list[j]}_weight_ih_l{i}"] = (
                getattr(model, f"weight_ih_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )[
                0
            ]  # Add the [0], to conform to black formatting, but not store in ()

            r[f"r__{weight_type_list[j]}_weight_hh_l{i}"] = (
                getattr(model, f"weight_hh_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )[0]

            bi[f"bi_{weight_type_list[j]}_bias_ih_l{i}"] = (
                getattr(model, f"bias_ih_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )[0]

            bh[f"bh_{weight_type_list[j]}_bias_hh_l{i}"] = (
                getattr(model, f"bias_hh_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )[0]

    # store all weight in one dict, call theta in line with paper tolga
    theta = dict({"w": w, "r": r, "bi": bi, "bh": bh})

    return theta


def get_gradient_weights(model, hidden_dim, theta_gradients=None):
    """
    Returns the weight ordered as in the paper(see Tolga)
    Using the scheme below and the knowledge that the weights in the paper (see Tolga, anomaly) correspond as the following:
    W(x)=Wix, R(x)=Whx and b(x) = bix + bhx, where x is of {I,f,z/g,o}. Where z=g respectively.


    LSTM.weight_ih_l[k] – the learnable input-hidden weights of the \text{k}^{th}k th
    layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is (4*hidden_size, num_directions * hidden_size).
    If proj_size > 0 was specified, the shape will be (4*hidden_size, num_directions * proj_size) for k > 0

    ~LSTM.weight_hh_l[k] – the learnable hidden-hidden weights of the \text{k}^{th}k th
     layer (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size). If proj_size > 0 was specified, the shape will be (4*hidden_size, proj_size).

    ~LSTM.bias_ih_l[k] – the learnable input-hidden bias of the \text{k}^{th}kthlayer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)

    ~LSTM.bias_hh_l[k] – the learnable hidden-hidden bias of the \text{k}^{th}kth
    layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)
    source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    """
    # Lists that make it easy to select the matching weight matrixes that are stored in one tensor/matrix by pytorch model
    weight_type_list = ["i", "f", "g", "o"]
    weight_type_selector = [k * hidden_dim for k in [0, 1, 2, 3]]
    weight_type_selector.append(None)

    # Coresponds to W,R and B respectively:
    w = dict()
    r = dict()
    bi = dict()
    bh = dict()

    if not theta_gradients:
        # Loop over the total present layers
        for i in range(model.num_layers):
            # loop over all weight types
            for j in range(4):
                # Make sure all names have the same string length between beginning and first {}
                w[f"w__{weight_type_list[j]}_weight_ih_l{i}"] = (
                    getattr(model, f"weight_ih_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[
                    0
                ]  # Add the [0], to conform to black formatting, but not store in ()

                r[f"r__{weight_type_list[j]}_weight_hh_l{i}"] = (
                    getattr(model, f"weight_hh_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0]

                bi[f"bi_{weight_type_list[j]}_bias_ih_l{i}"] = (
                    getattr(model, f"bias_ih_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0]

                bh[f"bh_{weight_type_list[j]}_bias_hh_l{i}"] = (
                    getattr(model, f"bias_hh_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0]
    else:
        # Loop over the total present layers
        for i in range(model.num_layers):
            # loop over all weight types
            for j in range(4):
                # Make sure all names have the same string length between beginning and first {}
                # Add the [0], to conform to black formatting, but not store in ()
                w[f"w__{weight_type_list[j]}_weight_ih_l{i}"] = (
                    getattr(model, f"weight_ih_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0] + theta_gradients["w"][f"w__{weight_type_list[j]}_weight_ih_l{i}"]

                r[f"r__{weight_type_list[j]}_weight_hh_l{i}"] = (
                    getattr(model, f"weight_hh_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0] + theta_gradients["r"][f"r__{weight_type_list[j]}_weight_hh_l{i}"]

                bi[f"bi_{weight_type_list[j]}_bias_ih_l{i}"] = (
                    getattr(model, f"bias_ih_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0] + theta_gradients["bi"][f"bi_{weight_type_list[j]}_bias_ih_l{i}"]

                bh[f"bh_{weight_type_list[j]}_bias_hh_l{i}"] = (
                    getattr(model, f"bias_hh_l{i}").grad[
                        weight_type_selector[j] : weight_type_selector[j + 1]
                    ],
                )[0] + theta_gradients["bh"][f"bh_{weight_type_list[j]}_bias_hh_l{i}"]

    # store all weight in one dict, call theta in line with paper tolga
    theta_gradients = dict({"w": w, "r": r, "bi": bi, "bh": bh})

    return theta_gradients


def put_weight_in_pytorch_matrix(weight, weight_name, pytorch_weights):
    """
    Function puts the (updated) weight at the correction position in the pytorch weights structure
    """

    batch_size = len(weight)
    if weight_name[3] == "i":
        pytorch_weights[:batch_size] = weight
    elif weight_name[3] == "f":
        pytorch_weights[batch_size : batch_size * 2] = weight
    elif weight_name[3] == "g":
        pytorch_weights[batch_size * 2 : batch_size * 3] = weight
    else:
        pytorch_weights[batch_size * 3 :] = weight

    return pytorch_weights


def get_full_pytorch_weight(weights, device):
    """
    # get full original weights, called pytorch_weights (following lstm structure)
    weights: dict for specified weight group (see get_weights in functions/data_manipulation for dict type)
    Store the information per lstm layer, i.e. 1st layer is 0 2nd layer is 1 etc.
    """
    pytorch_weights = dict()

    track_layer = None
    for weight_name, weight in weights.items():
        # check layer, and thus if new temp is needed
        if weight_name[-1] != str(track_layer):
            temp = torch.empty(0, device=device)
            track_layer = weight_name[-1]

        temp = torch.cat((temp, weight))  # Double () is needed

        # Check if it is the final layer number
        if len(temp) == 4 * len(weight):
            pytorch_weights[weight_name[-1]] = temp

    return pytorch_weights


def h_bar_list_to_numpy(h_bar_list, device=torch.device("cpu")):
    """Function that converts a list type object filled
    with h_bar's to a numpy object. If device was cuda,
    the data is returned to device: cpu first.

    Args:
        h_bar_list (list): Contains h_bar objects
        device (torch.device): Currently used device (cpu/cuda)

    Returns:
        numpy.Array: Numpy array containing h_bar objects
    """
    if device.type == "cpu":
        # still make sure that the device is indeed cpu
        h_bar_list = h_bar_list.cpu()
        h_bar_list_np = np.array(
            [h_bar.detach().numpy() for h_bar in h_bar_list], dtype=object
        )
    else:
        h_bar_list_temp = h_bar_list.to(device=torch.device("cpu"))
        h_bar_list_np = np.array([h_bar.detach().numpy() for h_bar in h_bar_list_temp])

    return h_bar_list_np


def scaled_epsilon_n_max_epochs(learning_rate):
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

    epsilon = 10 ** -(6 + order_of_magnitude)

    more_epochs = 100 * (order_of_magnitude - 3) if order_of_magnitude > 3 else 0
    max_epochs = 400 + more_epochs  # order_of_magnitude * 50

    return epsilon, max_epochs


def separate_anomalies_from_regular(anomaly_track, jets_index, data: list):
    """
    Note data must be the filtered data returning with the same length as the anomaly_track list
    return dict, if passed
    """
    # select correct order of jets to match with anomalies
    data = [data[i] for i in jets_index]

    if len(anomaly_track) != len(data):
        return "Failed"

    anomalies = ak.Array(list(compress(data, anomaly_track == -1)))
    non_anomalies = ak.Array(list(compress(data, anomaly_track == 1)))

    return anomalies, non_anomalies


def trials_df_and_minimum(trials_results, test_param="loss"):
    """Function that takes a dictionary of trials and converts
    this to a Pandas Dataframe. Subsequently the minimum loss
    is taken and the model(s) corresponding to this loss is/are
    selected and their information is printed.

    Args:
        trials_results (dict): Dictionary containing the trials results
        test_param (str, optional): Parameter which was tested. Can be loss or cost
                                    or something else. Defaults to "loss".

    Returns:
        Tuple[Pandas.Dataframe, float, Pandas.Dataframe, dict_keys]:
            - Pandas Dataframe of the trials results
            - Minimum test_param value
            - Pandas Dataframe of trials corresponding to minimum test_param
            - List of hyperparameter names
    """
    # reform to complete list of trials
    try:
        trials_list = [trial for trial in trials_results["_trials"]]
    except:
        trials_list = [
            trial
            for trials in [trials["_trials"] for trials in trials_results]
            for trial in trials
        ]
    parameters = trials_list[0]["result"]["hyper_parameters"].keys()

    # build DataFrame
    df = pd.concat(
        [pd.json_normalize(trial["result"]) for trial in trials_list]
    ).reset_index()
    df = df[df["loss"] != 10]  # filter out bad model results

    # get minima
    min_val = df[test_param].min()
    min_idxs = df.index[df[test_param] == min_val].to_list()
    min_df = df[df[test_param] == min_val].reset_index()

    # print best model(s) hyperparameters:
    print("\nBest Hyper Parameters:")
    hyper_parameters_df = min_df.loc[
        :, min_df.columns.str.startswith("hyper_parameters")
    ]
    for index, row in hyper_parameters_df.iterrows():
        print(f"\nModel {index} from trial {min_idxs[index]}:")
        for key in hyper_parameters_df.keys():
            print("  {:12}\t  {}".format(key.split(".")[1], row[key]))
        print(f"with loss: \t\t{min_df['loss'].iloc[index]}")
        print(f"with final cost:\t{min_df['final_cost'].iloc[index]}")

    return df, min_val, min_df, parameters


def cut_on_length(data, length, features=[na.recur_jetpt, na.recur_dr, na.recur_z]):
    """Select jets from recursive set with only specific jet length"""
    a = []
    for i in range(len(data)):
        if len(data[i][features[0]]) == length:
            a.append(data[i])
    a = ak.Array(a)
    return a
