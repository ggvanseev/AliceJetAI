"""
Make file to generate cost condition and cost plots from a list of job ids.
"""

import torch
from plotting.cost_condition import cost_condition_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = ["11857498", "2023_04_15"]  # example

# select "test" or "train"
trial_type = "test"

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_{trial_type}_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# create cost condition plots from trials and jobs
for job_id, trials in zip(job_ids, trials_test_list):
    cost_condition_plots(trials, job_id)
