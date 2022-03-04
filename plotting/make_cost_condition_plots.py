import torch
import time
import matplotlib.pyplot as plt
import os

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "9783665",
    "9783675",
    "9783688",
]

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device)
    for job_id in job_ids
]

for job_id, trials in zip(job_ids, trials_test_list):

    out_dir = f"output/cost_condition_{job_id}"
    out_txt = ""

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    for i, trial in enumerate(trials["_trials"]):
        out_txt += f"trial: {i}"

        # obtain results from the trial
        result = trial["result"]

        # extract hyper parameters from the results
        h_parm = result["hyper_parameters"]
        title_plot = f""
        for key in h_parm:
            title_plot += f"{h_parm[key]}_{key}_"
            out_txt += "\n  {:12}\t  {}".format(key, h_parm[key])

        # extract cost data from the results
        cost_data = result["cost_data"]
        if cost_data != 10: # check for failed models
            track_cost = cost_data["cost"]
            track_cost_condition = cost_data["cost_condition"]
        else:
            out_txt += "\n *** FAILED MODEL *** \n"
            break

        # plot cost condition and cost function
        fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
        fig.suptitle(title_plot, y=1.08)
        ax1.plot(track_cost_condition[1:], label="Cost Condition")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Cost Condition")

        ax2 = ax1.twinx()
        ax2.plot(track_cost[1:], "--", linewidth=0.5, alpha=0.7, label="Cost")
        ax2.set_ylabel("Cost")

        fig.legend()
        fig.savefig(out_dir + "/" f"trial_{i}.png")
        plt.close(fig)  # close figure - clean memory
        out_txt += "\n\n"
    
    txt_file = open(out_dir+"/info.txt", "w")
    txt_file.write(out_txt)
    txt_file.close()
    
