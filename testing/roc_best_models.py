import time
import torch
from functions.data_loader import load_n_filter_data, load_n_filter_data_qg, mix_quark_gluon_samples, load_trials
from plotting.stacked import *
from plotting.roc import *
from functions.data_manipulation import (
    get_y_results_from_trial,
    get_y_results_from_trial_h,
    separate_anomalies_from_regular,
    cut_on_length,
    train_dev_test_split,
)

### ----- User Input ----- ###
# obtain best_dict from roc_auc_scores.py
best_dict = {'11120653': 3, '22_07_18_1520': 0, '22_08_11_1520': (0, 0), 'sigJetRecur_dr12': 0} # hand cut lstm: (trial nr, hidden dim nr.)

# Setup to make multiple roc plot
labels = ["LSTM + OCSVM - HyperTraining", "LSTM + OCSVM", "Hand Cut LSTM Hidden State", "Hand Cut Variables"]
colors = ["C1", "C2", "C3", "C4"]

# file setup
out_files = [] # leave empty if no specific file to use # you can load your premade mix here: pickled file
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root" # if available
file_name = "samples/JetToyHIResultSoftDropSkinny.root"

# setup as in get_anomalies
g_percentage = 90
num = 0 # trial nr.
save_flag = True
show_distribution_percentages_flag = False

pre_made = False   # created in regular training
mix = True        # set to true if mixture of q and g is required
kt_cut = None         # for dataset, splittings kt > 1.0 GeV, assign None if not using
### ---------------------- ###


# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### Regular Training options ### TODO make this automatic 
# get dr_cut as in the regular training!
dr_cut = None # np.linspace(0,0.4,len(job_ids)+1)[i+1]

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
g_recur = split_test_data_recur[split_test_data[na.parton_match_id] == 21]
q_recur = split_test_data_recur[abs(split_test_data[na.parton_match_id]) < 7]

# mock arrays for moniker 1 or 0 if gluon or quark
g_true = ak.Array([{"y_true": 1} for i in range(len(g_recur))])
q_true = ak.Array([{"y_true": 0} for i in range(len(q_recur))])

# mix 90% g vs 10% q of 1500
data_list = [{**item, **y} for item, y in zip(g_recur.to_list(), g_true.to_list())] + [{**item, **y} for item, y in zip(q_recur.to_list(), q_true.to_list())]

print("Loading data complete")       
    
 # variable list - store for input
variables = g_recur.fields

# set font size
plt.rcParams.update({'font.size': 13.5})

# make supplots
fig, ax = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
ax.plot([0,1],[0,1],color='k')
    
# First plot
job_id, trial = list(best_dict.items())[0]
trials = load_trials(job_id, remove_unwanted=True) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial(data_list, trials[trial])
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc:.2f}")
ax.plot(fpr, tpr, color=colors[0], label=labels[0]+"\n"+f" AUC: {roc_auc:.2F}")     

# Second plot
job_id, trial = list(best_dict.items())[1]
trials = load_trials(job_id, remove_unwanted=False) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial(data_list, trials[trial])
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc:.2f}")
ax.plot(fpr, tpr, color=colors[1], label=labels[1]+"\n"+f" AUC: {roc_auc:.2F}") 

# Third plot - Hand cut on lstm hidden dims
job_id, (trial, dim) = list(best_dict.items())[2]
trials = load_trials(job_id, remove_unwanted=False) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial_h(data_list, trials[trial], dim)
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc:.2f}")
ax.plot(fpr, tpr, color=colors[2], label=labels[2]+"\n"+f" AUC: {roc_auc:.2F}") 

# Fourth plot - Hand cut on variable
job_id, trial = list(best_dict.items())[3]
y_predict = [d[job_id][0] for d  in data_list]
y_true = [d['y_true'] for d in data_list]
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc:.2f}")
ax.plot(fpr, tpr, color=colors[3], label=labels[3]+"\n"+f" AUC: {roc_auc:.2F}") 
ax.set_xlabel("Normal Fraction Quarks")
ax.set_ylabel("Normal Fraction Gluons")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.4)
ax.legend() #TODO

# store roc curve plots in designated directory
out_dir = f"output/ROC_curves_best"
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

out_file = out_dir + "/roc_best_" + time.strftime("%y_%m_%d_%H%M")

# save plot without title
plt.savefig(out_file+"_no_title")

# save plot with title
plt.title("Best Models By AUC", y=1.04)
plt.savefig(out_file)
        
    
