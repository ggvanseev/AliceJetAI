"""
Make file to generate cost condition and cost plots from a list of job ids.
"""

import torch
from plotting.cost_condition import cost_condition_plots

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    # "22_07_18_1327",
    # "22_07_18_1334",
    # "22_07_18_1340",
    # "22_07_18_1345",
    # "22_07_18_1348",
    # "22_07_18_1357",
    # "22_07_18_1404",
    # "22_07_18_1410",
    # "22_07_18_1414",
    # "22_07_18_1417",
    # "22_07_18_1424",
    # "22_07_18_1432",
    # "22_07_18_1435",
    # "22_07_18_1440",
    # "22_07_18_1445",
    # "22_07_18_1452",
    # "22_07_18_1502",
    # "22_07_18_1508",
    # "22_07_18_1514",
    # "22_07_18_1520",
    # "22_08_09_1909",
    # "22_08_09_1934",
    # "22_08_09_1939",
    # "22_08_09_1941",
    # "22_08_11_1520",
    # "22_08_12_1302",
    # "22_08_12_1306",
    # "22_08_12_1311",
    # "22_08_12_1313",
    "22_09_05_1618",
    "22_09_06_1613",
    "10993304",
    "10993305",
    "11120653",
    "11120654",
    "11120655",
    "11316966",
    "11316967",
    "11461549",
    "11461550",
    "11474168",
    "11474169",
    "11478120",
    "11478121",
    "11487531", 
    "11503919", 
    "11503920",
    "11487532", 
    "11503917", 
    "11503918",
    "11852650",
    "11852651",
    # "11852652",
    # "11852653",
    "11857497",
    "11857498",
]
# job_ids = ["11852650"]

# select "test" or "train"
trial_type = "test" # TODO what is this even?

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_{trial_type}_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# create cost condition plots from trials and jobs
for job_id, trials in zip(job_ids, trials_test_list):
    cost_condition_plots(trials, job_id)
