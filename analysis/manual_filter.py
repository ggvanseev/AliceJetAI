import torch


job_id = 10206558
selected_list = [0, 1]


# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = torch.load(
    f"storing_results/trials_test_{job_id}.p", map_location=device
)

# select only desired
selected_trials = dict()
# Add weird nesting for consitency without manual fitler to not adjust code (no further meaning)
selected_trials["_trials"] = dict()

for i, j in enumerate(selected_list):
    selected_trials["_trials"][i] = trials_test_list["_trials"][j]

out_file = f"storing_results/trials_test_manual_filter_{job_id}.p"

torch.save(selected_list, open(out_file, "wb"))
