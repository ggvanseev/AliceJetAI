"""
Make file to generate violin plots from a list of job ids.
"""

import torch
from functions.data_manipulation import trials_df_and_minimum
from plotting.violin import violin_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "10993302",
    "10993303",
    "11120653",
    "11120654",
    "11120655",
# ]
# job_ids = [
    "11316965",
    "11316966",
    "11316965",
]

# select test parameter: e.g. "loss" or "final_cost"
test_param = "final_cost"

# load trials results from file and store in dict with their job_id
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = {
    job_id : torch.load(
        f"storing_results/trials_test_{job_id}.p", map_location=device
    )
    for job_id in job_ids
}

# convert trials to Dataframe and get minimum/minima
df, min_val, min_df, parameters = trials_df_and_minimum(trials_test_list, test_param)

# create violin plots which are stored in "output/violin_plots_{job_id}"
violin_plots(df, min_val, min_df, parameters, job_ids, test_param)
