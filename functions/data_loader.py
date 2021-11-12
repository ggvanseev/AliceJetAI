import uproot
import awkward as ak
import pandas as pd
import numpy as np

import names as na


def Samples(file_name: str, treeN, cut: bool = True):
    branches = uproot.open(file_name)["jetTreeSig"].arrays()
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

    # OPTIONAL: apply cuts, cut only on jetpt
    if cut:
        eta_cut = 2
        pt_cut = 130
        g_jets_recur = g_recur_jets[
            (g_jets[na.jet_eta] <= eta_cut)
            & (g_jets[na.jet_eta] >= -eta_cut)
            & (g_jets[na.jetpt] >= pt_cut)
        ]
        q_recur_jets = q_recur_jets[
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

    """
    # This is the same code but with awkward arrays 
    jets = branches2[['sigJetPt', 'sigJetEta', 'sigJetPhi', 'sigJetM', 'sigJetArea', 'sigJetRecur_nSD', 'jetInitPDG']]
    jetRecs = branches2[['sigJetRecur_dr12', 'sigJetRecur_jetpt', 'sigJetRecur_z']]

    gjets = jets[jets['jetInitPDG'] == 21 ]
    qjets = jets[(jets['jetInitPDG'] >= -6) & (jets['jetInitPDG'] <= 6)]

    cgjets = gjets[(gjets['sigJetEta'] <= 2) & (gjets['sigJetEta'] >= -2) & (gjets['sigJetPt'] >= 130)]
    cqjets = qjets[(qjets['sigJetEta'] <= 2) & (qjets['sigJetEta'] >= -2) & (qjets['sigJetPt'] >= 130)]

    gRecjets = jetRecs[jets['jetInitPDG'] == 21 ]
    qRecjets = jetRecs[(jets['jetInitPDG'] >= -6) & (jets['jetInitPDG'] <= 6)]

    cgRecjets = gRecjets[(gjets['sigJetEta'] <= 2) & (gjets['sigJetEta'] >= -2) & (gjets['sigJetPt'] >= 130)]
    cqRecjets = qRecjets[(qjets['sigJetEta'] <= 2) & (qjets['sigJetEta'] >= -2) & (qjets['sigJetPt'] >= 130)]
    """

    return g_jets, q_jets, g_recur_jets, q_recur_jets
