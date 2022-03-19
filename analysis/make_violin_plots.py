import torch
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functions.data_manipulation import trials_df_and_minimum
from plotting.general import violin_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "9756505",
]

# select test parameter: e.g. "loss" or "final_cost"
test_param = "loss"

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device)
    for job_id in job_ids
]

df, min_val, min_df, parameters = trials_df_and_minimum(trials_test_list, test_param)
violin_plots(df, min_val, min_df, parameters, job_ids, test_param)
