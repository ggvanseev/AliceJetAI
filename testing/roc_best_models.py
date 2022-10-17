import time
import torch
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

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
# obtain best_dict from roc_auc_scores.py - NOTE second dict is with OCSVM only as well
best_dict = {'LSTM + OCSVM - HyperTraining': ('11120653', 11), 'LSTM + OCSVM - RegularTraining': ('11478121', 1), 'Cut And Count $R_g$': ('sigJetRecur_dr12', 0)} #  'Hand Cut LSTM Hidden State': ('11478121', (7, 0)),
best_dict = {'LSTM + OCSVM - HyperTraining': ('11120653', 11), 'LSTM + OCSVM - RegularTraining': ('11478121', 1),  r'OCSVM $\nu=0.1$ - First Splittings': ('last_reversed_nu0.1', 0), 'Cut And Count $R_g$ - First Splittings': ('sigJetRecur_dr12', 0)} # 'Hand Cut LSTM Hidden State': ('11478121', (7, 0)),
# best_dict = {'LSTM + OCSVM - HyperTraining': ('11120653', 11), 'LSTM + OCSVM - RegularTraining': ('11461550', 7),  r'OCSVM $\nu=0.1$ - First Splittings': ('last_reversed_nu0.1', 0), 'Cut And Count $R_g$ - First Splittings': ('sigJetRecur_dr12', 0)} # 'Hand Cut LSTM Hidden State': ('11478121', (7, 0)),


# Setup to make multiple roc plot
colors = ["C1", "C2", "C3", "C4", "C5"]
colors = sns.color_palette()

# file setup
out_files = [] # leave empty if no specific file to use # you can load your premade mix here: pickled file
file_name = "/data/alice/wesselr/JetToyHIResultSoftDropSkinny_100k.root" # if available
file_name = "samples/JetToyHIResultSoftDropSkinny_100k.root"

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
split_train_data_recur, _, split_test_data_recur = train_dev_test_split(jets_recur, split=[0.7, 0.1])
_,  _, split_test_data = train_dev_test_split(jets, split=[0.7, 0.1])
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
    
# First plot - Hypertraining
label, (job_id, trial) = list(best_dict.items())[0]
trials = load_trials(job_id, remove_unwanted=False) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial(data_list, trials[trial])
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc}")
ax.plot(fpr, tpr, color=colors[0], label=label+"\n"+f" AUC: {roc_auc:.4F}")     

# Second plot - Regular Training
label, (job_id, trial) = list(best_dict.items())[1]
trials = load_trials(job_id, remove_unwanted=False) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial(data_list, trials[trial])
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc}")
ax.plot(fpr, tpr, color=colors[1], label=label+"\n"+f" AUC: {roc_auc:.4F}") 

# Third plot - Hand cut on lstm hidden dims
""" NOTE leave this out
label, (job_id, (trial,dim)) = list(best_dict.items())[2]
trials = load_trials(job_id, remove_unwanted=False) # do not remove unwanted, otherwise trial nr. is wrong
y_true, y_predict = get_y_results_from_trial_h(data_list, trials[trial], dim)
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc}")
ax.plot(fpr, tpr, color=colors[2], label=label+"\n"+f" AUC: {roc_auc:.4F}") 
"""

# Fourth plot - OCSVM only
# prepare data
scaler = MinMaxScaler()
split_train_data = format_ak_to_list(split_train_data_recur)
split_dev_data = format_ak_to_list(split_test_data_recur)
scaler.fit([s for jet in split_train_data for s in jet]) # fit as single branch
train_data = [scaler.transform(d) for d in split_train_data] # then transform it
dev_data = [scaler.transform(d) for d in split_dev_data] # then transform it
train_data = [train_data[i][0] for i in range(len(train_data))] # last reversed pooling
dev_data = [dev_data[i][0] for i in range(len(dev_data))] # last reversed pooling

# now use ocsvm to fit and predict
# make and fit model
svm_model = OneClassSVM(nu=0.1, gamma = "auto", kernel="linear")
svm_model.fit(train_data)
y_predict = svm_model.decision_function(dev_data) 
y_true = [1 if jet[na.parton_match_id] == 21 else 0 for jet in split_test_data]

# make plot
label, (job_id, trial) = list(best_dict.items())[2]
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc}")
ax.plot(fpr, tpr, color=colors[2], label=label+"\n"+f" AUC: {roc_auc:.4F}") 
ax.set_xlabel("Normal Fraction Quarks")
ax.set_ylabel("Normal Fraction Gluons")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.4)
ax.legend() #TODO

# Fifth plot - Hand cut on variable
label, (job_id, trial) = list(best_dict.items())[3]
y_predict = [d[job_id][0] for d  in data_list]
y_true = [d['y_true'] for d in data_list]
fpr, tpr, _ = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print(f"ROC Area under curve: {roc_auc}")
ax.plot(fpr, tpr, color=colors[3], label=label+"\n"+f" AUC: {roc_auc:.4F}") 
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
plt.close('all')        
    
