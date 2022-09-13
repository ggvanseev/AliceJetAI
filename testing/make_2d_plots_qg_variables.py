
import torch

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