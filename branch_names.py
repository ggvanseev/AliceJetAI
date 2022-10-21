# tree
tree = "jetTreeSig;1"
# After appyling (recursive) softdrop
recur_kts = "sigJetRecur_kts"
recur_dr = "sigJetRecur_dr12"
recur_tf = "sigJetRecur_tf"
recur_jetpt = "sigJetRecur_jetpt"
recur_omega = "sigJetRecur_omegas"
recur_splits = "sigJetRecur_nSD"
recur_z = "sigJetRecur_z"

# After applying a clustering algorithm
jetpt = "sigJetPt"
jet_eta = "sigJetEta"
jet_phi = "sigJetPhi"
jet_M = "sigJetM"
jet_area = "sigJetArea"

# variable names from JetToyHI monickers
variable_names = {
    "sigJetRecur_kts" : r"$k_T$",
    "sigJetRecur_dr12" : r"$R_g$",
    "sigJetRecur_tf" : "$t_f$",
    "sigJetRecur_jetpt" : r"$p_T$",
    "sigJetRecur_omegas" : "$\omega_s$",
    "sigJetRecur_nSD" : "Nr. SD Splittings",
    "sigJetRecur_z" : r"Recursive $z$",
    "sigJetPt" : r"$Jet p_T$",
    "sigJetEta" : r"$Jet \eta$",
    "sigJetPhi" : r"$Jet \phi$",
    "sigJetM" : "Jet Mass",
    "sigJetArea" : "Jet Area",
}

# TODO: also CA, but different names, discuss which ./runSoftDrop to use
parton_match_id = "jetInitPDG"
test = 1

# Bas variables
tau1 = "tau1"
tau2 = "tau2"
tau2tau1 = "tau2tau1"
z2_theta1 = "z2_theta1"
z2_theta15 = "z2_theta15"

# plotting labels
sigma = r"$\sigma$"
axis_n_devide_njets_bandwith = r"$dN/(N_{jets}d_{bandwidth})$"
