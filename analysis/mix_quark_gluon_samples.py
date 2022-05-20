"""
Little tool to mix sample data sets of quark and gluon jets
"""

import uproot
import torch
import awkward as ak

from functions.data_loader import load_n_filter_data, load_n_filter_data_qg
import branch_names as na



# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_500k.root"
# file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"

kt_cut = False

# percentage of gluon jets, q jets will
g_percentage = 90


# --- Program ---
# open file and select jet data and recursive jet data
data = load_n_filter_data_qg(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id] ,kt_cut=kt_cut)

total = round(len(data[0])/100) # judge total off of about the total number of gluon jets
g_len = total * g_percentage
q_len = total *(100 - g_percentage) 

# train, test, val splits as in regular_training
split = [0.7, 0.1]

torch.save(
    ak.concatenate((data[0][:g_len], data[1][:q_len])),
    f"samples/mixed_{total*100}jets_pct:{g_percentage}g_{100-g_percentage}q.p",
)
