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
import io

# list of ids of jobs to be merged
job_ids = [
    9577887,
    9577887,
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# contents = pickle.load(f) becomes...
file = torch.load(
    f"storing_results/trials_test_{9577887}.burrell.nikhef.nl.p", map_location=device,
)
