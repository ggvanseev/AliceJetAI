import torch
import time
import matplotlib.pyplot as plt
import os

from plotting.general import cost_condition_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "17_03_22_1717",
]

# select "test" or "train"
trial_type = "train"

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_{trial_type}_{job_id}.p", map_location=device)
    for job_id in job_ids
]

for job_id, trials in zip(job_ids, trials_test_list):
    cost_condition_plots(trials, job_id)
    
