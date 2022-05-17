import torch
from PIL import Image
import numpy as np
import win32gui as wg
from win32gui import GetForegroundWindow
import win32com.client

job_ids = [
    "10240832",
    "10240834",
    "10240835",
    "10240836",
    "10240837",
    "10240838",
    "10240839",
    "10240840",
    "10240841",
]
# If list left empty, will give visual options using terminal
selected_list = []


# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device)
    for job_id in job_ids
]
for job_id, trials in zip(job_ids, trials_test_list):
    # give user the option to select using the terminal
    if len(selected_list) < 1:
        n_figures = len(trials["_trials"])

        # This gets the details of the current window, the one running the program
        aw = np.zeros(n_figures + 1)
        aw[0] = GetForegroundWindow()
        shell = win32com.client.Dispatch("WScript.Shell")

        for i in range(n_figures):
            with Image.open(f"output/cost_condition_{job_id}/trial_{i}.png") as img:
                img.show()
                continue_flag = True
                while continue_flag:
                    shell.SendKeys("%")
                    wg.SetForegroundWindow(aw[i])
                    get_pass = input("y or n:")
                    if get_pass:
                        aw[i + 1] = GetForegroundWindow()
                    if get_pass == "y":
                        selected_list.append(i)
                        continue_flag = False
                    elif get_pass == "n":
                        continue_flag = False

    # select only desired
    selected_trials = dict()

    # store selection with respect to original
    selected_trials["original_index"] = selected_list
    # save selected trials
    selected_trials["_trials"] = list()

    for i, j in enumerate(selected_list):
        selected_trials["_trials"].append(trials["_trials"][j])

    out_file = f"storing_results/manual_selected/trials_test_manual_filter_{job_id}.p"

    torch.save(selected_trials, open(out_file, "wb"))

    selected_list = list()