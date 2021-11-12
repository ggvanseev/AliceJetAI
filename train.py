from typing import no_type_check
import torch
import torch.nn as nn
import numpy as np

import uproot
import awkward as ak
import pandas as pd

import names as na
from ai import *
from functions import dataSamples

# load root dataset into a pandas dataframe
fileName = './samples/JetToyHIResultSoftDropSkinny.root'
g_jets, q_jets, g_recur_jets, q_recur_jets = dataSamples.Samples(fileName)

print(g_jets.head(), len(g_jets))
print(g_recur_jets.head(), len(g_recur_jets))

# check if gpu is available, otherwise use cpu
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def train():
    return
