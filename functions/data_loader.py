"""
Loading in data from a ROOT file using uproot. 

load_n_filter data will give the gluon and quark data from SoftDropped and Recursive 
SoftDropped dataset of specified ROOT file. Data is later filtered according to jet 
initiator PDG, eta and pt cuts. Output is in four Awkward arrays: two for quark and
gluon jet data and two for their recursive counterparts.


"""
import uproot
import awkward as ak
import pandas as pd
import numpy as np
from typing import Tuple

import names as na


def load_n_filter_data(
    file_name: str,
    tree_name: str = "jetTreeSig",
    cut: bool = True,
    eta_cut: float = 2.0,
    pt_cut: int = 130,
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """Load in dataset from ROOT file of jet data. Subsequently, the jet data will be
    split into sets for quarks and gluons as well as sets with recursive jet data for
    quarks and gluons. Therefore, the provided dataset is required to contain the 
    variables: jetpt, jet_eta, jet_phi, jet_M, jet_area, recur_splits & parton_match_id.
    Exact labels are given in 'names.py'.

    Args:
        file_name (str): File name of ROOT dataset.
        tree_name (str, optional): Name of the TTree object of the ROOT file. Defaults
                to "jetTreeSig".
        cut (bool, optional): Boolean statement whether to apply cuts. Defaults to True.
        eta_cut (float, optional): Value for eta cut. Defaults to 2.0.
        pt_cut (int, optional): Value for pt cut. Defaults to 130.
        
    Returns:
        Tuple[ak.Array, ak.Array, ak.Array, ak.Array]: Tuple containing all gluon and
                quark datasets.
    """

    # open file and select jet data and recursive jet data
    branches = uproot.open(file_name)[tree_name].arrays()
    jets = branches[
        [
            na.jetpt,
            na.jet_eta,
            na.jet_phi,
            na.jet_M,
            na.jet_area,
            na.recur_splits,
            na.parton_match_id,
        ]
    ]
    jets_recur = branches[[na.recur_dr, na.recur_jetpt, na.recur_z]]

    # select quark and gluon jet data
    g_jets = jets[jets[na.parton_match_id] == 21]
    q_jets = jets[abs(jets[na.parton_match_id] < 7)]

    # select recursive quark and gluon jet data
    g_jets_recur = jets_recur[jets[na.parton_match_id] == 21]
    q_jets_recur = jets_recur[abs(jets[na.parton_match_id] < 7)]

    # apply cuts, default: -2 < eta < 2 and jet_pt > 130 GeV
    if cut:
        g_jets_recur = g_jets_recur[
            (abs(g_jets[na.jet_eta]) < eta_cut) & (g_jets[na.jetpt] > pt_cut)
        ]
        q_jets_recur = q_jets_recur[
            (abs(q_jets[na.jet_eta]) < eta_cut) & (q_jets[na.jetpt] > pt_cut)
        ]
        g_jets = g_jets[
            (abs(g_jets[na.jet_eta]) < eta_cut) & (g_jets[na.jetpt] > pt_cut)
        ]
        q_jets_ = q_jets[
            (abs(q_jets[na.jet_eta]) < eta_cut) & (q_jets[na.jetpt] > pt_cut)
        ]

    return g_jets, q_jets, g_jets_recur, q_jets_recur


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
