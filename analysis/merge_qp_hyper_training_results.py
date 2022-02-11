from hyperopt import fmin, tpe, hp, Trials, trials_from_docs
import itertools
import torch
import pickle
import io

# list of ids of jobs to be merged
job_ids = [
    "10_02_22_first",
    "10_02_22",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# contents = pickle.load(f) becomes...
trials_lists = [
    list(torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device,))
    for job_id in job_ids
]

trial1 = torch.load(
    f"storing_results/trials_test_10_02_22_first.p", map_location=device,
)
trial2 = torch.load(f"storing_results/trials_test_10_02_22.p", map_location=device,)
trials_merged = trials_from_docs(list(trial1) + list(trial2))
a = 0
