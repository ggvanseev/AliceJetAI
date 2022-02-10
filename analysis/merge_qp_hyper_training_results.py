from hyperopt import (
    fmin,
    tpe,
    hp,
    space_eval,
    STATUS_OK,
    Trials,
)
import torch
import pickle

job_id1 = 9577887
job_id2 = 9577887

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

file1 = pickle.load(
    open(f"storing_results/trials_test_{job_id1}.burrell.nikhef.nl.p", "rb")
)
file2 = pickle.load(
    open(f"storing_results/trials_test_{job_id2}.burrell.nikhef.nl.p", "rb")
)
