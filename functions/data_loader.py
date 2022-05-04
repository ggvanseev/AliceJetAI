"""
Loading in data from a ROOT file using uproot. 

load_n_filter data will give the gluon and quark data from SoftDropped and Recursive 
SoftDropped dataset of specified ROOT file. Data is later filtered according to jet 
initiator PDG, eta and pt cuts. Output is in four Awkward arrays: two for quark and
gluon jet data and two for their recursive counterparts.


"""
import uproot
import awkward as ak
import branch_names as na
from typing import Tuple
import branch_names as na
import numpy as np
import random


def select_non_empty_branches(branches, non_empty_key):
    """
    non_empty_key: give along a recur, because due to zcut a branch might have become empty
    """
    # First filter all completely empty branches
    branch_reference = branches[non_empty_key]
    non_empty = list()
    for i in range(len(branch_reference)):
        if len(branch_reference[i]) > 0 and ak.any(branch_reference[i]):
            non_empty.append(i)

    branches = branches[non_empty]
    branch_reference = branches[non_empty_key]

    # empty partially empty entries
    mask = ak.ArrayBuilder()
    for i in range(len(branch_reference)):
        mask.begin_list()
        for j in range(len(branch_reference[i])):
            if len(branch_reference[i, j]) > 0 and ak.any(branch_reference[i, j]):
                mask.integer(j)
        mask.end_list()

    for field in branches.fields:
        branches[field] = branches[field][mask]
    return branches


def flatten_array(branches, step_size=2000):
    """
    returns a flattend array
    """
    new_branches = dict()

    for field in branches.fields:
        # Use this work around to avoid memory issues with ak.concatenate.
        temp_array = ak.ArrayBuilder()
        n_steps = len(branches[field]) // step_size + 1
        steps = np.append(np.arange(n_steps) * step_size, None)
        for i in np.arange(n_steps):
            temp_array.append(
                ak.concatenate(branches[field][steps[i] : steps[i + 1]], axis=0)
            )
        new_branches[field] = ak.flatten(temp_array, axis=1)

    return ak.Array(new_branches)


def load_n_filter_data_single(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    kt_cut: float = None,
    eta_max: float = 2.0,
    pt_min: int = 130,
    jet_recur_branches: list = [na.recur_dr, na.recur_jetpt, na.recur_z],
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]: 
   
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
        ]
    ]
    jets_recur = branches[jet_recur_branches]
    
    # Print some info on dataset.
    print(f"Number of jets in dataset:\t\t{np.count_nonzero(jets[na.jetpt])}")

    # apply cuts: -2 < eta < 2 and jet_pt > 130 GeV
    if cut:
        print("Applying cuts: -2 < eta < 2 and jet_pt > 130 GeV")
        jets_recur = jets_recur[
            (abs(jets[na.jet_eta]) < eta_max) & (jets[na.jetpt] > pt_min)
        ]
        jets = jets[
            (abs(jets[na.jet_eta]) < eta_max) & (jets[na.jetpt] > pt_min)
        ]
        print(
            f"\tjets left after cuts:\t{np.count_nonzero(jets[na.jetpt])}"
        )

    # apply kt_cut of kt > 1.0 GeV
    if kt_cut:
        print(f"Applying cut: kt > {kt_cut} GeV on all splittings")
        # get kt values
        jets_kt = (
            jets_recur.sigJetRecur_jetpt
            * jets_recur.sigJetRecur_dr12
            * jets_recur.sigJetRecur_z
        )

        # cut kts
        jets_recur = jets_recur[jets_kt > kt_cut]
        print(f"\tsplittings cut:\t\t{1 - np.count_nonzero(jets_kt[jets_kt > kt_cut]) / np.count_nonzero(jets_kt):.2%}")


        # hist gluons TODO keep for possible later analysis: histograms
        #g_kts_flat = ak.flatten(ak.flatten(g_jets_kt)).to_list()
        # g_kts_hist = np.histogram(g_kts_flat, bins=range(round(max(g_kts_flat)))) 

        # hist quarks TODO keep for possible later analysis: histograms
        #q_kts_flat = ak.flatten(ak.flatten(q_jets_kt)).to_list()
        # q_kts_hist = np.histogram(q_kts_flat, bins=range(round(max(q_kts_flat)))) 

    # remove empty additions from recursive jets and flatten them, i.e. take jet out of event nesting
    jets_recur = select_non_empty_branches(
        jets_recur, non_empty_key=jet_recur_branches[0]
    )

    jets_recur = flatten_array(jets_recur)

    return jets_recur



def load_n_filter_data(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    kt_cut: float = None,
    eta_max: float = 2.0,
    pt_min: int = 130,
    jet_recur_branches: list = [na.recur_dr, na.recur_jetpt, na.recur_z],
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
        kt_cut (float, optional): Functions as boolean statement whether to apply kt_cut
                to splittings by setting a value. Value for kt cut measured in GeV, sets
                a minumum. Defaults to None, i.e. no kt_cut.
        eta_max (float, optional): Value for eta cut. Defaults to 2.0.
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
    jets_recur = branches[jet_recur_branches]

    # Print some info on dataset. Note: Nr of jets is significantly larger than nr of quark/gluon jets.
    # This is because we only know for sure which jets are quark or gluon jets from the Parton Initiator,
    # which in turn means that we can at most obtain 1 quark or gluon jet per event.
    print(f"Number of jets in dataset:\t\t{np.count_nonzero(jets[na.jetpt])}")
    print(
        f"Number of gluon jets in dataset:\t{np.count_nonzero(jets[na.parton_match_id] == 21)}"
    )
    print(
        f"Number of quark jets in dataset:\t{np.count_nonzero(abs(jets[na.parton_match_id]) < 7)}"
    )

    # select quark and gluon jet data
    g_jets = jets[jets[na.parton_match_id] == 21]
    q_jets = jets[abs(jets[na.parton_match_id]) < 7]

    # select recursive quark and gluon jet data
    g_jets_recur = jets_recur[jets[na.parton_match_id] == 21]
    q_jets_recur = jets_recur[abs(jets[na.parton_match_id]) < 7]

    # apply cuts: -2 < eta < 2 and jet_pt > 130 GeV
    if cut:
        print("Applying cuts: -2 < eta < 2 and jet_pt > 130 GeV")
        g_jets_recur = g_jets_recur[
            (abs(g_jets[na.jet_eta]) < eta_max) & (g_jets[na.jetpt] > pt_min)
        ]
        q_jets_recur = q_jets_recur[
            (abs(q_jets[na.jet_eta]) < eta_max) & (q_jets[na.jetpt] > pt_min)
        ]
        g_jets = g_jets[
            (abs(g_jets[na.jet_eta]) < eta_max) & (g_jets[na.jetpt] > pt_min)
        ]
        q_jets = q_jets[
            (abs(q_jets[na.jet_eta]) < eta_max) & (q_jets[na.jetpt] > pt_min)
        ]
        print(
            f"\tgluon jets left after cuts:\t{np.count_nonzero(g_jets[na.parton_match_id] == 21)}"
        )
        print(
            f"\tquark jets left after cuts:\t{np.count_nonzero(abs(q_jets[na.parton_match_id] < 7))} "
        )

    # apply kt_cut of kt > 1.0 GeV
    if kt_cut:
        print(f"Applying cut: kt > {kt_cut} GeV on all splittings")
        # get kt values
        g_jets_kt = (
            g_jets_recur.sigJetRecur_jetpt
            * g_jets_recur.sigJetRecur_dr12
            * g_jets_recur.sigJetRecur_z
        )
        q_jets_kt = (
            q_jets_recur.sigJetRecur_jetpt
            * q_jets_recur.sigJetRecur_dr12
            * q_jets_recur.sigJetRecur_z
        )

        # cut kts
        g_jets_recur = g_jets_recur[g_jets_kt > kt_cut]
        q_jets_recur = q_jets_recur[q_jets_kt > kt_cut]
        print(
            f"\tgluon splittings cut:\t\t{1 - np.count_nonzero(g_jets_kt[g_jets_kt > kt_cut]) / np.count_nonzero(g_jets_kt):.2%}"
        )
        print(
            f"\tquark splittings cut:\t\t{1 - np.count_nonzero(q_jets_kt[q_jets_kt > kt_cut]) / np.count_nonzero(q_jets_kt):.2%}"
        )

        # hist gluons TODO keep for possible later analysis: histograms
        # g_kts_flat = ak.flatten(ak.flatten(g_jets_kt)).to_list()
        # g_kts_hist = np.histogram(g_kts_flat, bins=range(round(max(g_kts_flat))))

        # hist quarks TODO keep for possible later analysis: histograms
        # q_kts_flat = ak.flatten(ak.flatten(q_jets_kt)).to_list()
        # q_kts_hist = np.histogram(q_kts_flat, bins=range(round(max(q_kts_flat))))

    # remove empty additions from recursive jets and flatten them, i.e. take jet out of event nesting
    g_jets_recur = select_non_empty_branches(
        g_jets_recur, non_empty_key=jet_recur_branches[0]
    )

    g_jets_recur = flatten_array(g_jets_recur)

    q_jets_recur = select_non_empty_branches(
        q_jets_recur, non_empty_key=jet_recur_branches[0]
    )

    q_jets_recur = flatten_array(q_jets_recur)

    return g_jets_recur, q_jets_recur
