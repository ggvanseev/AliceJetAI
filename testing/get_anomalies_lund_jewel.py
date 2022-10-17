from lib2to3.pgen2.literals import simple_escapes
import torch
from functions.classification import get_anomalies
from functions.data_loader import *
from functions.data_manipulation import cut_on_length, separate_anomalies_from_regular, train_dev_test_split
import matplotlib.pyplot as plt

from testing.plotting_test import lund_planes_anomalies, lund_planes_anomalies_qg, lund_planes_qg, normal_vs_anomaly_2D_qg

file_name = "samples/SDTiny_jewelNR_120_vac-1.root"
file_name = "samples/SDTiny_jewelNR_120_simple-1.root"

job_ids = [
    # 11542141, # vac
    # 11542142, # vac
    11542143, # simple
    11542143, # simple
    11542144, # simple
    11542144, # simple
]
trial_nrs = [
    # 3, # vac
    # 1, # vac
    8, # simple
    9, # simple
    6, # simple
    9, # simple
]

kt_cut = None
dr_cut = None

# set current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load and filter data for criteria eta and jetpt_cap
# You can load your premade mix here: pickled file w q/g mix
jets_recur, _ = load_n_filter_data(file_name, kt_cut=kt_cut, dr_cut=dr_cut)
_, _, split_test_data_recur = train_dev_test_split(jets_recur, split=[0.7, 0.1])
print("Loading data complete")  


for i, (job_id, num) in enumerate(zip(job_ids, trial_nrs)):
    
    print(f"\nAnomalies run: {i+1}, job_id: {job_id}") # , for dr_cut: {dr_cut}")
    ###--------------------------###
    
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(f"No succesful trial for job: {job_id}. Try to complete a new training with same settings.")
        continue
    print("Loading trials complete")
    
    type_jets = "Jewel "
    _, jets_index_tracker, classification_tracker = get_anomalies(split_test_data_recur, job_id, trials, file_name, jet_info=type_jets)
     # get anomalies for trial: num
    anomaly, normal = separate_anomalies_from_regular(
        anomaly_track=classification_tracker[num],
        jets_index=jets_index_tracker[num],
        data=jets_recur,
    )
    
    lund_planes_anomalies(normal, anomaly, job_id, num)
