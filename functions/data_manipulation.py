import torch
import torch.nn as nn
import awkward as ak
import numpy as np


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


def train_validation_split(dataset, split=0.8):
    """
    Split is the percentage cut of for selecting training data.
    Thus split 0.8 means 80% of data is considered for training.
    """
    max_train_index = int(len(dataset) * split)
    train_data = dataset[0:max_train_index]
    validation_data = dataset[max_train_index:]

    return train_data, validation_data
