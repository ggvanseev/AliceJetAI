"""
Make file to generate svm boundary plots from a list of job ids.
"""

import torch
from plotting.svm_boundary import svm_boundary_plots
from testing.testing_functions import load_digits_data

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "22_04_28_1438",
]
# select "test" or "train"
trial_type = "test"

# fit for testing?
fit=True

# ----- data that was trained on: -----
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
#print('Mixed "9": 675 = 90% of normal data with "0": 75 = 10% as anomalous data for a train set of 750 samples')
test_data = test_dict["9"][:360] + test_dict["0"][:40]
#print('Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples')

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_{trial_type}_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# create cost condition plots from trials and jobs 
for job_id, trials in zip(job_ids, trials_test_list):
    svm_boundary_plots(trials, job_id, train_data, fit=fit)
    
