import uproot
import awkward as ak
import pandas as pd
import numpy as np

import names as na


def Samples(fileName: str, cut=True):
    branches = uproot.open(fileName)['jetTreeSig'].arrays()
    jet_df = ak.to_pandas(branches[[na.jetpt, na.jet_eta, na.jet_phi,
                                    na.jet_M, na.jet_area, na.recur_splits, na.parton_match_id]])
    jet_recur_df = ak.to_pandas(
        branches[[na.recur_dr, na.recur_jetpt, na.recur_z]])

    # quark and gluon jet data
    g_jets = jet_df[jet_df[na.parton_match_id] == 21]
    q_jets = jet_df[(jet_df[na.parton_match_id] >= -6) &
                    (jet_df[na.parton_match_id] <= 6)]

    # recursive quark and gluon jet data
    g_recur_jets = jet_recur_df[jet_df[na.parton_match_id] == 21]
    q_recur_jets = jet_recur_df[(
        jet_df[na.parton_match_id] >= -6) & (jet_df[na.parton_match_id] <= 6)]

    # OPTIONAL: apply cuts, cut only on jetpt
    if cut:
        eta_cut = 2
        pt_cut = 130
        g_recur_jets = g_recur_jets[(g_jets[na.jet_eta] <= eta_cut) & (
            g_jets[na.jet_eta] >= -eta_cut) & (g_jets[na.jetpt] >= pt_cut)]
        q_recur_jets = q_recur_jets[(q_jets[na.jet_eta] <= eta_cut) & (
            q_jets[na.jet_eta] >= -eta_cut) & (q_jets[na.jetpt] >= pt_cut)]
        g_jets = g_jets[(g_jets[na.jet_eta] <= eta_cut) & (
            g_jets[na.jet_eta] >= -eta_cut) & (g_jets[na.jetpt] >= pt_cut)]
        q_jets_ = q_jets[(q_jets[na.jet_eta] <= eta_cut) & (
            q_jets[na.jet_eta] >= -eta_cut) & (q_jets[na.jetpt] >= pt_cut)]

    return g_jets, q_jets, g_recur_jets, q_recur_jets
