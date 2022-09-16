
import torch
from functions.classification import get_anomalies
from functions.data_loader import *
from functions.data_manipulation import cut_on_length, separate_anomalies_from_regular, train_dev_test_split
import matplotlib.pyplot as plt

from testing.plotting_test import lund_planes_anomalies, lund_planes_qg, normal_vs_anomaly_2D_qg

# file_name(s) - comment/uncomment when switching between local/Nikhef
#file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root"
file_name = "samples/JetToyHIResultSoftDropSkinny.root"

job_ids = [
    "11461549",
    "11461550"           
]

g_percentage = 50 # for evaluation of stacked plots 50%, ROC would be nice to have 90 vs 10 percent
num = 0 # trial nr.
save_flag = True
show_distribution_percentages_flag = False

pre_made = False   # created in regular training
mix = True        # set to true if mixture of q and g is required
kt_cut = None         # for dataset, splittings kt > 1.0 GeV, assign None if not using


out_files=[]

# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# collect all auc values from ROC curves
all_aucs = {}

for i, job_id in enumerate(job_ids):
    
    # start from a point in the series
    # if i < 0:
    #   continue
    
    ### Regular Training options ### TODO make this automatic 
    # get dr_cut as in the regular training!
    dr_cut = None# np.linspace(0,0.4,len(job_ids)+1)[i+1] 
    
    print(f"\nAnomalies run: {i+1}, job_id: {job_id}") # , for dr_cut: {dr_cut}")
    ###--------------------------###
    
   # you can load your premade mix here: pickled file
    # Load and filter data for criteria eta and jetpt_cap
    # You can load your premade mix here: pickled file w q/g mix
    if out_files:
        jets_recur, jets = torch.load(file_name)
    elif mix:
        jets_recur, jets, file_name_mixed_sample = mix_quark_gluon_samples(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], g_percentage=g_percentage, kt_cut=kt_cut, dr_cut=dr_cut)
    else:
        jets_recur, _ = load_n_filter_data(file_name, jet_branches=[na.jetpt, na.jet_M, na.parton_match_id], kt_cut=kt_cut, dr_cut=dr_cut)
    
    # split data TODO see if it works -> test set too small for small dataset!!! -> using full set
    _, split_test_data_recur, _ = train_dev_test_split(jets_recur, split=[0.0, 1.0])
    _, split_test_data, _ = train_dev_test_split(jets, split=[0.0, 1.0])
    # split_test_data_recur = jets_recur
    # split_test_data= jets
    
    # # split data into quark and gluon jets
    g_jets_recur = split_test_data_recur[split_test_data[na.parton_match_id] == 21]
    q_jets_recur = split_test_data_recur[abs(split_test_data[na.parton_match_id]) < 7]
    
    print("Loading data complete")       
    
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(f"No succesful trial for job: {job_id}. Try to complete a new training with same settings.")
        continue
    print("Loading trials complete")
    
    # gluon jets, get anomalies and normal out
    type_jets = "g_jets"
    _, g_jets_index_tracker, g_classification_tracker = get_anomalies(g_jets_recur, job_id, trials, file_name, jet_info=type_jets)

    # quark jets, get anomalies and normal out
    type_jets = "q_jets"
    _, q_jets_index_tracker, q_classification_tracker = get_anomalies(q_jets_recur, job_id, trials, file_name, jet_info=type_jets)
    

    if show_distribution_percentages_flag:
        plt.figure(f"Distribution histogram anomalies {jet_info}", figsize=[1.36 * 8, 8])
        plt.hist(anomalies_info["percentage_anomalies"])
        plt.xlabel(f"Percentage (%) jets anomalies {jet_info}")
        plt.ylabel(f"N")

    features = [na.recur_jetpt, na.recur_dr, na.recur_z]

    for num in range(len(trials)):
        # get anomalies for trial: num
        g_anomaly, g_normal = separate_anomalies_from_regular(
            anomaly_track=g_classification_tracker[num],
            jets_index=g_jets_index_tracker[num],
            data=g_jets_recur,
        )
        q_anomaly, q_normal = separate_anomalies_from_regular(
            anomaly_track=q_classification_tracker[num],
            jets_index=q_jets_index_tracker[num],
            data=q_jets_recur,
        )
        
        # other selection/cuts
        extra_cuts = False
        if extra_cuts == True:
            
            # cut on length
            length = 3
            g_anomaly = cut_on_length(g_anomaly, length, features)
            g_normal = cut_on_length(g_normal, length, features)
            q_anomaly = cut_on_length(q_anomaly, length, features)
            q_normal = cut_on_length(q_normal, length, features)
            job_id += f"_jet_len_{length}"
            
            pass
        for splittings in ['all', 'first', 'last', 'mean']:
            normal_vs_anomaly_2D_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, splittings, job_id, num)
        lund_planes_anomalies(g_anomaly, g_normal, q_anomaly, q_normal, job_id, num)
    lund_planes_qg(g_anomaly, g_normal, q_anomaly, q_normal, job_id)