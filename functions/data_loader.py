import uproot
import branch_names as na


def load_n_filter_data(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    eta_cut=2,
    pt_cut=130,
    jet_recur_branches: list = [na.recur_dr, na.recur_jetpt, na.recur_z],
):
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

    # quark and gluon jet data
    g_jets = jets[jets[na.parton_match_id] == 21]
    q_jets = jets[abs(jets[na.parton_match_id]) <= 6]

    # recursive quark and gluon jet data
    g_recur_jets = jets_recur[jets[na.parton_match_id] == 21]
    q_recur_jets = jets_recur[abs(jets[na.parton_match_id]) <= 6]

    # OPTIONAL: apply cuts, cut only on jetpt
    if cut:
        g_recur_jets = g_recur_jets[
            (abs(g_jets[na.jet_eta]) <= eta_cut) & (g_jets[na.jetpt] >= pt_cut)
        ]
        q_recur_jets = q_recur_jets[
            (abs(q_jets[na.jet_eta]) <= eta_cut) & (q_jets[na.jetpt] >= pt_cut)
        ]
        g_jets = g_jets[
            (abs(g_jets[na.jet_eta]) <= eta_cut) & (g_jets[na.jetpt] >= pt_cut)
        ]
        q_jets = q_jets[
            (abs(q_jets[na.jet_eta]) <= eta_cut) & (q_jets[na.jetpt] >= pt_cut)
        ]

    return g_jets, q_jets, g_recur_jets, q_recur_jets
