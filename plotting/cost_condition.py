"""
Contains all the cost condition and cost functions used in plotting.
"""

import matplotlib.pyplot as plt
import os

def cost_condition_plot(result: dict, title_plot: str, out_file: str):
    """Generate a cost vs cost condition plot from a trial result.
    The result should contain the "cost_data" as how this is stored
    in the training, existing of cost and cost condition data. 

    Args:
        result (dict): A single result from a dict of trials
        title_plot (str): Title that is given to the current plot
        out_file (str): Name/location of out file

    Returns:
        int: -1 is bad model
    """    
    
    # extract cost data from the results
    cost_data = result["cost_data"]
    if cost_data != 10:  # check for failed models
        track_cost = cost_data["cost"]
        track_cost_condition = cost_data["cost_condition"]
    else:
        out_txt += "\n *** FAILED MODEL *** \n"
        return -1 # No success

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

    # save and close the plot
    fig.savefig(out_file)
    plt.close(fig)  # close figure - clean memory


def cost_condition_plots(trials: dict, job_id):
    """Generates all cost vs cost condition plots from trials.
    Trials object must be converted to dict as is done in the
    training.

    Args:
        trials (dict): dictionary made of Trials object
        job_id (str): Id given to this training/run
    """    

    # make out directory if it does not exist yet
    out_dir = f"output/cost_condition_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # store trial parameters
    out_txt = ""

    # build plots for each trial
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

        # add final loss and cost to the info
        out_txt += f"\n{'with loss:':18}{result['loss']}"
        out_txt += f"\n{'with final cost:':18}{result['final_cost']}"
        
        # generate the plot
        fig = cost_condition_plot(result, title_plot, out_file=out_dir+ "/" f"trial_{i}.png")
        if fig == -1:
            break

        out_txt += "\n\n"

    # save info on all trials
    txt_file = open(out_dir + "/info.txt", "w")
    txt_file.write(out_txt)
    txt_file.close()

