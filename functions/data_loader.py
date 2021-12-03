"""
Loading in data from a ROOT file using uproot. 

load_n_filter data will give the gluon and quark data from SoftDropped and Recursive 
SoftDropped dataset of specified ROOT file. Data is later filtered according to jet 
initiator PDG, eta and pt cuts. Output is in four Awkward arrays: two for quark and
gluon jet data and two for their recursive counterparts.


"""
import uproot
import awkward as ak
import names as na
from typing import Tuple
import branch_names as na


def load_n_filter_data(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    eta_cut: float = 2.0,
    pt_cut: int = 130,
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
    jets_recur = branches[jet_recur_branches]

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
