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


def load_n_filter_data(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    kt_cut: bool = True,
    eta_max: float = 2.0,
    pt_min: int = 130,
    kt_min: float = 1.0,
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
        kt_cut (bool, optional): Boolean statement whether to apply kt_cut to splittings. Defaults to True.
        eta_max (float, optional): Value for eta cut. Defaults to 2.0.
        kt_min (float, optional): Value for kt cut. Defaults to 1.0 GeV.
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
    print(f"Number of gluon jets in dataset:\t{np.count_nonzero(jets[na.parton_match_id] == 21)}")
    print(f"Number of quark jets in dataset:\t{np.count_nonzero(abs(jets[na.parton_match_id] < 7))}")

    # select quark and gluon jet data
    g_jets = jets[jets[na.parton_match_id] == 21]
    q_jets = jets[abs(jets[na.parton_match_id] < 7)]

    # select recursive quark and gluon jet data
    g_jets_recur = jets_recur[jets[na.parton_match_id] == 21]
    q_jets_recur = jets_recur[abs(jets[na.parton_match_id] < 7)]

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
        print(f"\tgluon jets left after cuts:\t{np.count_nonzero(g_jets[na.parton_match_id] == 21)}")
        print(f"\tquark jets left after cuts:\t{np.count_nonzero(abs(q_jets[na.parton_match_id] < 7))} ")
        
    # apply kt_cut of kt > 1.0 GeV
    if kt_cut:
        print("Applying cut: kt > 1.0 GeV on all splittings")
        # get kt values
        g_jets_kt = g_jets_recur.sigJetRecur_jetpt * g_jets_recur.sigJetRecur_dr12 * g_jets_recur.sigJetRecur_z
        q_jets_kt = q_jets_recur.sigJetRecur_jetpt * q_jets_recur.sigJetRecur_dr12 * q_jets_recur.sigJetRecur_z
        
        # cut kts
        g_jets_recur = g_jets_recur[g_jets_kt > kt_min]
        q_jets_recur = q_jets_recur[q_jets_kt > kt_min]
        
        # hist gluons
        g_kts_flat = ak.flatten(ak.flatten(g_jets_kt)).to_list()
        g_kts_hist = np.histogram(g_kts_flat, bins=range(round(max(g_kts_flat))))
        print(f"\tgluon splittings cut:\t\t{g_kts_hist[0][0] / sum(g_kts_hist[0]):.2%}")
        
        # hist quarks
        q_kts_flat = ak.flatten(ak.flatten(q_jets_kt)).to_list()
        q_kts_hist = np.histogram(q_kts_flat, bins=range(round(max(q_kts_flat))))
        print(f"\tquark splittings cut:\t\t{q_kts_hist[0][0] / sum(q_kts_hist[0]):.2%}")
    
    return g_jets, q_jets, g_jets_recur, q_jets_recur
