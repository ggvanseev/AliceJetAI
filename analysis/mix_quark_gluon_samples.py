"""
Little tool to mix sample data sets of quark and gluon jets
"""

from isort import file
import uproot
import torch
import awkward as ak
import numpy as np
import random

from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, mix_quark_gluon_samples
import branch_names as na


# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"


###------------------###
### Multiple samples ###
###------------------###
runs = 20
for i in range(runs):
    
    # setup for this specific run
    dr_cut = np.linspace(0,0.4,runs+1)[i+1]
    
    jets_recur, jets, out_file = mix_quark_gluon_samples(file_name=file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], dr_cut=dr_cut)

