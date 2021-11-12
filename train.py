from typing import no_type_check

import torch
import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

import names as na
from ai import *
from functions import data_loader

# load root dataset into a pandas dataframe
fileName = "./samples/JetToyHIResultSoftDropSkinny.root"
g_jets, q_jets, g_recur_jets, q_recur_jets = data_loader.load_n_filter_data(fileName)

print(ak.to_pandas(g_jets).head(), len(ak.to_pandas(g_jets)), sep="\n")
print(ak.to_pandas(g_recur_jets).head(), len(ak.to_pandas(g_recur_jets)), sep="\n")

# check if gpu is available, otherwise use cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def train():
    return
