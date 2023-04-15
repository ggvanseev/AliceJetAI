"""
Load a dataset and test a few things to see what is in it.
"""

import uproot
import awkward as ak
import matplotlib.pyplot as plt

# specify path to filename
# file_name = "samples/SDTiny_jewelNR_120_simple-1.root"
file_name = "samples/JetToyHIResultSoftDropTiny_zc01.root"
# print(os.getcwd()) # used to check your working directory

# obtain tree/branches from the file
file = uproot.open(file_name)
tree = file["jetTreeSig"]
branches = tree.arrays()[
    ["sigJetRecur_dr12", "sigJetRecur_jetpt", "sigJetRecur_z"]
]  # 'branches' now stores the observables in 'awkward arrays'

r_g = ak.flatten(branches["sigJetRecur_dr12"]).to_list()
r_g = [x for x in r_g if x]
firsts = [jet[0] for jet in r_g]

plt.figure()
plt.hist(firsts, bins=60)
plt.title("First Splittings")
plt.ylabel("Count")
plt.xlabel("$R_g$")
plt.show()

plt.figure()
plt.hist(ak.flatten(ak.flatten(branches["sigJetRecur_dr12"])), bins=60)
plt.title("All Splittings")
plt.ylabel("Count")
plt.xlabel("$R_g$")
plt.show()
