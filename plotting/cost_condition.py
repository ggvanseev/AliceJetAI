"""
Contains all the cost condition and cost functions used in plotting.
"""

#import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pylab as plt

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
        
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 13.5})
    
    # extract cost data from the results
    cost_data = result["cost_data"]
    if cost_data != 10:  # check for failed models
        track_cost = cost_data["cost"]
        track_cost_condition = cost_data["cost_condition"]
    else:
        out_txt += "\n *** FAILED MODEL *** \n"
        return -1, out_txt # No success

    # plot cost condition and cost function
    fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
    cost_con = ax1.plot(track_cost_condition[1:], linewidth=1.3,alpha=0.85, label="Cost Condition:\n"+r"$(\kappa(\mathbf{ \theta }_{k+1}, \mathbf{ \alpha }_{k+1}) - \kappa(\mathbf{ \theta }_k, \mathbf{ \alpha }_k))^2$")
    ax1.set_xlabel(r"Epoch: $k$")
    ax1.set_ylabel(r"Cost Condition")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=len(track_cost_condition[1:]))

    ax2 = ax1.twinx()
    cost = ax2.plot(track_cost[1:], color="red", linewidth=1.3, alpha=0.85, label="Cost:\n"+r"$\kappa( \mathbf{ \theta }_{k+1}, \mathbf{ \alpha }_{k+1})$")
    ax2.set_ylabel(r"Cost")
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0, right=len(track_cost_condition[1:]))
    
    # figure setup
    legend = cost + cost_con
    labels = [l.get_label() for l in legend]
    ax2.legend(legend, labels, loc=0) # , borderaxespad=0.1 # put on ax2 since cost is more important -> legend will follow cost line
    ax2.grid(alpha=0.4) # add grid
    plt.tight_layout()
    #plt.subplots_adjust(left=0.1, bottom=0.2, right=0.87, top=0.86)

    # save version without title
    fig.savefig(out_file+"_no_title")
    fig.suptitle(title_plot) # save title afterwards
    
    # save and close the plot
    fig.savefig(out_file)
    plt.close(fig)  # close figure - clean memory
    return 


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
        title_plot = f"Training Results - Job: {job_id}"
        for key in h_parm:
            out_txt += "\n  {:12}\t  {}".format(key, h_parm[key])

        # add final loss and cost to the info
        out_txt += f"\n{'with loss:':18}{result['loss']}"
        out_txt += f"\n{'with final cost:':18}{result['final_cost']}"
                
        # generate the plot
        fig_out = cost_condition_plot(result, title_plot, out_file=out_dir+ "/" f"trial_{i}")
        if fig_out == -1:
            break
        out_txt += "\n\n"

    # save info on all trials
    txt_file = open(out_dir + "/info.txt", "w")
    txt_file.write(out_txt)
    txt_file.close()
    print("For "+job_id+":\n"+out_txt)

