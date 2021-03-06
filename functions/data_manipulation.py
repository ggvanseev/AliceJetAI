import torch
import torch.nn as nn
import awkward as ak
import numpy as np
from copy import copy
from torch.utils.data import TensorDataset, DataLoader


def collate_fn_pad(batch):
    """
    Padding of data
    """
    seq = [t[0] for t in batch]
    weight = [t[1] for t in batch]
    label = [t[2] for t in batch]
    length = [len(i) for i in seq]

    seq = nn.utils.rnn.pad_sequence(seq, batch_first=True)
    weight = torch.stack(weight)
    label = torch.stack(label)
    return seq, weight, label, length


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
    # remove empty entries and weird nestedness, e.g. dr[[...]], TODO
    lst = [[y[0] for y in x] for x in lst if x != [[], [], []]]
    # transpose remainder to get correct shape
    lst = [list(map(list, zip(*x))) for x in lst]
    return lst


def train_dev_test_split(dataset, split=[0.8, 0.1]):
    """
    Split is the percentage cut of for selecting training data.
    Thus split 0.8 means 80% of data is considered for training.
    """
    max_train_index = int(len(dataset) * split[0])
    max_dev_index = int(len(dataset) * (split[0] + split[1]))
    train_data = dataset[0:max_train_index]
    dev_data = dataset[max_train_index:max_dev_index]
    test_data = dataset[max_dev_index:]

    return train_data, dev_data, test_data


def branch_filler(dataset, batch_size, n_features=3, max_trials=100):
    # Count all values (, is indicative of a value), and devide by n_features to prevent double counting splits
    max_n_batches = int(str(dataset).count(",") / n_features / batch_size)

    # Batches, is a list with all the batches
    batches = []
    # Track_jets_in_batch tracks where the last split of the jet is located in the branch
    track_jets_in_batch = []

    for i in range(max_n_batches):

        # Space count tracks if branch is filled to max
        space_count = batch_size

        # make copies of the dataset to be able to remove elemnts while trying to fill branches
        # without destroyting original dataset
        temp_dataset = copy(dataset)
        temp_dataset2 = copy(dataset)

        # local trakcers of batches ad jets_in_batch
        batch = []
        jets_in_batch = []

        # variables that track progress
        add_branch_flag = True
        trials = 0

        # Check if batch is filled
        while space_count > 0:

            # loop over available jets
            for j in range(len(temp_dataset)):
                jet = temp_dataset[j]

                # Add jet to batch if possible
                if space_count >= len(jet):
                    batch.append(jet)
                    temp_dataset2.remove(jet)
                    space_count -= len(jet)
                    jets_in_batch.append(
                        batch_size - space_count - 1
                    )  # Position of the last split of jet

                # Track how many times it has looped over the datasets
                if j == len(temp_dataset) - 1:
                    temp_dataset = temp_dataset2
                    trials += 1

                if trials == max_trials:
                    add_branch_flag = False
                    dataset.remove(dataset[0])
                    space_count = 0

        if add_branch_flag:
            batches.append([y for x in batch for y in x])  # remove list nesting
            track_jets_in_batch.append(
                list(dict.fromkeys(jets_in_batch))
            )  # Only unique values (empty jet can add "end" jet)
            dataset = temp_dataset2

    # Remove list nesting of branches (will be restored by DataLoader from torch if same
    # batch size and shuffle=False is used)
    batches = [y for x in batches for y in x]

    return batches, track_jets_in_batch


def lstm_data_prep(*, data, scaler, batch_size, fit_flag=False):
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
    # test = TensorDataset(test_features, test_targets)

    return DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)


def get_weights(model, batch_size):
    """
    Returns the weight ordered as in the paper(see Tolga)
    Using the scheme below and the knowledge that the weights in the paper (see Tolga, anomaly) correspond as the following:
    W(x)=Wix, R(x)=Whx and b(x) = bix + bhx, where x is of {I,f,z/g,o}. Where z=g respectively.


    LSTM.weight_ih_l[k] ??? the learnable input-hidden weights of the \text{k}^{th}k th
    layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is (4*hidden_size, num_directions * hidden_size).
    If proj_size > 0 was specified, the shape will be (4*hidden_size, num_directions * proj_size) for k > 0

    ~LSTM.weight_hh_l[k] ??? the learnable hidden-hidden weights of the \text{k}^{th}k th
     layer (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size). If proj_size > 0 was specified, the shape will be (4*hidden_size, proj_size).

    ~LSTM.bias_ih_l[k] ??? the learnable input-hidden bias of the \text{k}^{th}kthlayer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)

    ~LSTM.bias_hh_l[k] ??? the learnable hidden-hidden bias of the \text{k}^{th}kth
    layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)
    source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    """
    # Lists that make it easy to select the matching weight matrixes that are stored in one tensor/matrix by pytorch model
    weight_type_list = ["i", "f", "g", "o"]
    weight_type_selector = [0, 1, 2, 3, 0] * batch_size

    # Coresponds to W,R and B respectively:
    w = dict()
    r = dict()
    b = dict()

    # Loop over the total present layers
    for i in range(model.num_layers):
        # loop over all weight types
        for j in range(4):
            w[f"{weight_type_list[j]}_{i}"] = (
                getattr(model, f"weight_ih_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )

            r[f"{weight_type_list[j]}_{i}"] = (
                getattr(model, f"weight_hh_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )

            b[f"{weight_type_list[j]}_{i}"] = (
                getattr(model, f"bias_ih_l{i}")[
                    weight_type_selector[j] : weight_type_selector[j + 1]
                ],
            )
            +getattr(model, f"bias_hh_l{i}")[
                weight_type_selector[j] : weight_type_selector[j + 1]
            ],

    return w, r, b
