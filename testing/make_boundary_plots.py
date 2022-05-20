"""
Make file to generate boundary plots from a list of job ids (digits data).
"""

import torch
from testing_functions import load_digits_data
from plotting.svm_boundary import svm_boundary_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = ["22_05_17_1047"]
# select "test" or "train"
trial_type = "test"

# ---- MAKE SURE THIS IS THE SAME AS IN THE TRAINING ----
# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"
file_name = train_file + "," + test_file

# get digits data
train_dict = load_digits_data(train_file)
test_dict = load_digits_data(test_file)

# mix "0" = 90% as normal data with "9" = 10% as anomalous data
train_data = train_dict["0"][:675] + train_dict["9"][75:150]

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_{trial_type}_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# create cost condition plots from trials and jobs
for job_id, trials in zip(job_ids, trials_test_list):
    svm_boundary_plots(trials, job_id, train_data)
