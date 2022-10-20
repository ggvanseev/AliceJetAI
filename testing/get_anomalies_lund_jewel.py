from lib2to3.pgen2.literals import simple_escapes
import torch
import awkward as ak
from functions.classification import get_anomalies
from functions.data_loader import *
from functions.data_manipulation import cut_on_length, separate_anomalies_from_regular, train_dev_test_split
import matplotlib.pyplot as plt
from plotting.stacked import stacked_plot, stacked_plot_sided, stacked_plot_sided_old

from testing.plotting_test import lund_planes
import branch_names as na

file_name_vac = "samples/SDTiny_jewelNR_120_vac-1.root"
file_name_simple = "samples/JetToyHIResultSoftDropTiny_zc01_simple-1.root"

job_ids = [
    "11542141", # vac
    "11542142", # vac
    "11852650", # simple
    "11852651", # simple
]
trial_nrs = [
    3, # vac
    1, # vac
    8, # simple
    9, # simple
]

kt_cut = None
dr_cut = None

# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load and filter data for criteria eta and jetpt_cap
# You can load your premade mix here: pickled file w q/g mix
jets_recur_vac, _ = load_n_filter_data(file_name_vac, kt_cut=kt_cut, dr_cut=dr_cut)
_, _, split_test_data_recur_vac = train_dev_test_split(jets_recur_vac, split=[0.7, 0.1])

jets_recur_simple, _ = load_n_filter_data(file_name_simple, kt_cut=kt_cut, dr_cut=dr_cut)
_, _, split_test_data_recur_simple = train_dev_test_split(jets_recur_simple, split=[0.7, 0.1])

# make same size of 3882 for current sets
split_test_data_recur_vac = split_test_data_recur_vac[:len(split_test_data_recur_simple)]
print(f"Vac dataset cut down to size {len(split_test_data_recur_vac)}, the size of simple test set")
print("Loading data complete")  

# lund plane vac versus simple
lund_planes(split_test_data_recur_vac, split_test_data_recur_simple, labels=["Unquenched Jets", "Contains Quenched Jets"], info="quenchedness")


for i, (job_id, num) in enumerate(zip(job_ids, trial_nrs)):
    
    print(f"\nAnomalies run: {i+1}, job_id: {job_id}") # , for dr_cut: {dr_cut}")
    ###--------------------------###
    
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(f"No succesful trial for job: {job_id}. Try to complete a new training with same settings.")
        continue
    print("Loading trials complete")
    
    type_jets = "Jewel"
    num = None
    
    # get anomalies
    _, jets_index_tracker_vac, classification_tracker_vac = get_anomalies(split_test_data_recur_vac, job_id, trials, file_name_vac, jet_info=type_jets+" Vac")
    _, jets_index_tracker_simple, classification_tracker_simple = get_anomalies(split_test_data_recur_simple, job_id, trials, file_name_simple, jet_info=type_jets+ " Simple")
    for num in ([num] if num is not None else range(len(trials))):
        # separate  anomalies for trial: num
        anomaly_vac, normal_vac = separate_anomalies_from_regular(
            anomaly_track=classification_tracker_vac[num],
            jets_index=jets_index_tracker_vac[num],
            data=split_test_data_recur_vac,
        )
        anomaly_simple, normal_simple = separate_anomalies_from_regular(
            anomaly_track=classification_tracker_simple[num],
            jets_index=jets_index_tracker_simple[num],
            data=split_test_data_recur_simple,
        )
            
        if i < 2: # first two on vac
            print("Using vac dataset for", job_id)
            lund_planes(normal_vac, anomaly_vac, job_id, trial=num)
        else:
            print("Using simple dataset for", job_id)
            lund_planes(anomaly_simple, normal_simple, job_id, trial=num)        
        
        for feature in [na.recur_jetpt, na.recur_dr, na.recur_z]:
            # store roc curve plots in designated directory
            out_dir = f"testing/output/stacked_{job_id}"
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass
        
            try:
                data = [[ak.firsts(normal_vac[feature]), ak.firsts(anomaly_vac[feature])], 
                        [ak.firsts(normal_simple[feature]), ak.firsts(anomaly_simple[feature])]]
                out_file =  out_dir + f"/trial{num}_first_" + feature 
                title = "Unquenched Samples Versus QGP Samples  - First Splitting"
                x_label = na.variable_names[feature]
                stacked_plot_sided_old(data, title, x_label, out_file, titles=["Unquenched Data", "Quenched Data"])
            except ValueError:
                print(f"Either no normal or no anomalous data for {job_id} trial {num}!")
                try:
                    a = ak.first(normal_vac[feature])
                except:
                    print("\tError in ak.first(normal_vac)")
                try:
                    a = ak.first(anomaly_vac[feature])
                except:
                    print("\tError in ak.first(anomaly_vac)")
                try:
                    a = ak.first(normal_simple[feature])
                except:
                    print("\tError in ak.first(normal_simple)")
                try:
                    a = ak.first(anomaly_simple[feature])
                except:
                    print("\tError in ak.first(anomaly_simple)")
