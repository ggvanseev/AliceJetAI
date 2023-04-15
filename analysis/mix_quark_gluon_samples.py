"""
Little tool to mix sample data sets of quark and gluon jets
and save them to /samples/mixed (this can also be done during
do_regular_training and do_hyper_training).
"""

import numpy as np

from functions.data_loader import (
    mix_quark_gluon_samples,
)
import branch_names as na


file_name = "samples/JetToyHIResultSoftDropSkinny.root"


###------------------###
### Multiple samples ###
###------------------###
runs = 20
for i in range(runs):

    # setup for this specific run
    dr_cut = np.linspace(0, 0.4, runs + 1)[i + 1]

    jets_recur, jets, out_file = mix_quark_gluon_samples(
        file_name=file_name,
        jet_branches=[na.jetpt, na.jet_M, na.parton_match_id],
        dr_cut=dr_cut,
    )
