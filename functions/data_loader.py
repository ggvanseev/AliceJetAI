import uproot
import awkward as ak
import pandas as pd
import numpy as np

import names as na


def load_n_filter_data(file_name: str, cut: bool = True, tree_name: str = "jetTreeSig"):
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

    # quark and gluon jet data
    g_jets = jets[jets[na.parton_match_id] == 21]
    q_jets = jets[(jets[na.parton_match_id] >= -6) & (jets[na.parton_match_id] <= 6)]

    # recursive quark and gluon jet data
    g_jets_recur = jets_recur[jets[na.parton_match_id] == 21]
    q_jets_recur = jets_recur[
        (jets[na.parton_match_id] >= -6) & (jets[na.parton_match_id] <= 6)
    ]

    # apply cuts: -2 < eta < 2 and jet_pt >= 130 GeV
    if cut:
        eta_cut = 2
        pt_cut = 130
        g_jets_recur = g_jets_recur[
            (g_jets[na.jet_eta] <= eta_cut)
            & (g_jets[na.jet_eta] >= -eta_cut)
            & (g_jets[na.jetpt] >= pt_cut)
        ]
        q_jets_recur = q_jets_recur[
            (q_jets[na.jet_eta] <= eta_cut)
            & (q_jets[na.jet_eta] >= -eta_cut)
            & (q_jets[na.jetpt] >= pt_cut)
        ]
        g_jets = g_jets[
            (g_jets[na.jet_eta] <= eta_cut)
            & (g_jets[na.jet_eta] >= -eta_cut)
            & (g_jets[na.jetpt] >= pt_cut)
        ]
        q_jets_ = q_jets[
            (q_jets[na.jet_eta] <= eta_cut)
            & (q_jets[na.jet_eta] >= -eta_cut)
            & (q_jets[na.jetpt] >= pt_cut)
        ]

    return g_jets, q_jets, g_jets_recur, q_jets_recur
