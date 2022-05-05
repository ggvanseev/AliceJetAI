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

# Properties of partons from pythia
parton_tf = "tfSplit"

# Time Clustering and Cambridge Aachen clustering
# After applying recursive softdrop and zcut=0.1
# Tc
# Parton matcher
tc_parton_match_id = "sigJetRecurZcutTc_partonMatchID"
tc_parton_match_dr = "sigJetRecurZcutTc_partonMatchDr"
# Of whole jet
tc_eta = "sigJetTcEta"
tc_pt = "sigJetTcPt"
# Of recursive
tc_recur_pt = "sigJetRecurZcutTc_jetpt"
tc_recur_tf = "sigJetRecurZcutTc_tf"
tc_recur_log_1dtheta = "sigJetRecurZcutTc_logdr12"
tc_recur_log_ztheta = "sigJetRecurZcutTc_logztheta"
tc_recur_omega = "sigJetRecurZcutTc_omegas"
tc_recur_zg = "sigJetRecurZcutTc_z"
# C/A
# Parton matcher
ca_parton_match_id = "sigJetRecurZcutCA_partonMatchID"
# Of whole jet
ca_eta = "sigJetCAEta"
ca_pt = "sigJetCAPt"
# Of recursive
ca_recur_pt = "sigJetRecurZcutCA_jetpt"
ca_recur_tf = "sigJetRecurZcutCA_tf"
ca_recur_log_1dtheta = "sigJetRecurZcutCA_logdr12"
ca_recur_log_ztheta = "sigJetRecurZcutCA_logztheta"
ca_recur_omega = "sigJetRecurZcutCA_omegas"
ca_recur_zg = "sigJetRecurZcutCA_z"

# TODO: also CA, but different names, discuss which ./runSoftDrop to use
parton_match_id = "jetInitPDG"
test = 1

# Bas variables
tau1 = "tau1"
tau2 = "tau2"
tau2tau1 = "tau2tau1"
z2_theta1 = "z2_theta1"
z2_theta15 = "z2_theta15"
