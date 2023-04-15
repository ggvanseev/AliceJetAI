"""
Load trial from a specific job and test a few things to see what is in it.
"""

import torch
from functions.data_loader import load_trials
from functions.data_manipulation import trials_df_and_minimum

# pick job & trial
job_id = "11120653"
trial_nr = 11

# load trials from job
trials = load_trials(job_id, remove_unwanted=False)
if not trials:
    print(
        f"No succesful trial for job: {job_id}. Try to complete a new training with same settings."
    )
print("Loading trials complete")

# set trial
trial = trials[trial_nr]
result = trial["result"]

# print info
print(f"\nFrom job {job_id} trial {trial}:")
for key, item in result["hyper_parameters"].items():
    print("  {:12}\t  {}".format(key, item))
print(f"with loss: \t\t{result['loss']}")
print(f"with final cost:\t{result['final_cost']}")
