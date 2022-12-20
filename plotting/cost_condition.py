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
    plt.rcParams.update({'font.size': 16})
    
    # extract cost data from the results
    cost_data = result["cost_data"]
    if cost_data != 10:  # check for failed models
        track_cost = cost_data["cost"]
        track_cost_condition = cost_data["cost_condition"]
    else:
        out_txt += "\n *** FAILED MODEL *** \n"
        return -1, out_txt # No success

    # plot cost condition function
    fig, ax1 = plt.subplots(sharex=True, figsize=[6 * 1.36, 6], dpi=160)
    final_cost = ax1.scatter(len(track_cost[1:])-1,track_cost[1:][-1], color="k", zorder=3, label=f"Final Cost: {track_cost[1:][-1]:.2E}")
    final_cost.set_clip_on(False) # so the marker can overlap the axis
    cost = ax1.plot([x for x in range(len(track_cost[1:]))], track_cost[1:], color="red", alpha=0.85, zorder=2, label="Cost:\n"+r"$\kappa( \mathbf{ \theta }_{k+1}, \mathbf{ \alpha }_{k+1})$")
    ax1.set_xlabel(r"Epoch $k$")
    ax1.set_ylabel(r"Cost")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=len(track_cost_condition[1:])-1)

    # plot cost function and final cost
    ax2 = ax1.twinx()
    cost_con = ax2.plot(track_cost_condition[1:], linewidth=1.3,alpha=0.85, zorder=1, label="Cost Condition:\n"+r"$(\kappa(\mathbf{ \theta }_{k+1}, \mathbf{ \alpha }_{k+1}) - \kappa(\mathbf{ \theta }_k, \mathbf{ \alpha }_k))^2$")
    cost_con[0].remove() # remove from this axis
    ax1.add_artist(cost_con[0]) # add to axis 1 to plot behind cost function
    ax2.set_ylabel(r"Cost Condition")
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0, right=len(track_cost_condition[1:])-1)
    
    # adjust for legend, at 40% of epochs
    epoch_halfway = int(len(track_cost[1:])* 2/5)
    max_cost_second_half = max(track_cost[epoch_halfway+1:])
    threshold =  0.6 * track_cost[0]
    if max_cost_second_half > threshold:
        factor = max_cost_second_half / threshold
        ax1.set_ylim(top=max(track_cost) * factor)
    
    # figure setup
    legend = cost + cost_con + [final_cost]
    labels = [l.get_label() for l in legend]
    plt.legend(legend, labels, loc=1) # , borderaxespad=0.1 # put on ax2 since cost is more important -> legend will follow cost line
    ax1.grid(axis="both", alpha=0.4) # add grid
    #plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9)

    # save version without title
    fig.savefig(out_file+"_no_title")
    fig.suptitle(title_plot) # save title afterwards
    
    # save and close the plot
    fig.savefig(out_file)
    plt.close(fig)  # close figure - clean memory
    return 


def cost_auc_plot(result: dict, title_plot: str, out_file: str):
    
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 16})
    
    # exctract cost & roc auc data
    track_roc_auc = result["cost_data"]["roc_auc"]
    track_cost = result["cost_data"]["cost"]
    
    #  figure setup & plot roc auc & final roc auc
    fig, ax = plt.subplots(sharex=True, figsize=[6 * 1.36, 6], dpi=160)
    cost = ax.plot(track_cost, zorder=2, alpha=0.85, color='r', label="Cost:\n"+r"$\kappa( \mathbf{ \theta }_{k+1}, \mathbf{ \alpha }_{k+1})$")
    ax.set_ylabel("Cost")    
    ax.set_xlabel("Epoch $k$")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=len(track_roc_auc[1:])-1)
    ax.grid( alpha=0.4)    
    
    # plot cost 
    ax1 = ax.twinx()
    roc = ax1.plot(track_roc_auc, zorder=1, alpha=0.85, label="ROC AUC\nOf Test Dataset")
    roc[0].remove() # remove from this axis
    ax.add_artist(roc[0]) # add to axis 1 to plot behind cost function
    final_roc = ax1.scatter(len(track_roc_auc[1:])-1,track_roc_auc[1:][-1], color="k", zorder=3, label=f"Final ROC AUC: {track_roc_auc[1:][-1]:.4f}")
    final_roc.set_clip_on(False) # so the marker can overlap the axis
    ax1.set_ylabel("Area Under Curve")
    ax1.set_ylim(bottom=0, top=1)
    ax1.set_xlim(left=0, right=len(track_roc_auc[1:])-1)
    
    # adjust for legend, at 40% of epochs
    epoch_halfway = int(len(track_cost[1:])* 2/5)
    max_cost_second_half = max(track_cost[epoch_halfway+1:])
    threshold =  0.6 * track_cost[0]
    if max_cost_second_half > threshold:
        factor = max_cost_second_half / threshold
        ax.set_ylim(top=max(track_cost) * factor)
        
    # adjust legend if auc track would pass through it    
    loc=1 # default
    bbox_loc = (1,1) # default
    mean_auc = sum(track_roc_auc[epoch_halfway+1:]) / len(track_roc_auc[epoch_halfway+1:])
    if mean_auc > 0.63:
        max_cost = max(track_cost)
        ax.set_ylim(top=max_cost * 1.05)
        bbox_loc = (1, mean_auc-0.03)  
        # check cost function, if more in the lower half, move legend to bottom left corner
        if len([x for x in track_cost if x >= max_cost*0.5]) > len([x for x in track_cost if x < max_cost*0.5]):
            bbox_loc = (0,0)
            loc = 3            
        
    # create legend & figure setup
    legend = cost + roc + [final_roc]
    labels = [l.get_label() for l in legend]
    ax1.legend(legend, labels, loc=loc, bbox_to_anchor=bbox_loc) # , borderaxespad=0.1 # put on ax2 since cost is more important -> legend will follow cost line
    plt.tight_layout()
    
    # save version without title
    fig.savefig(out_file+"_no_title")
    fig.suptitle(title_plot) # save title afterwards
    plt.tight_layout()
    
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

    # store trial parameters in out_txt
    out_txt = "For "+job_id+":"
    print(out_txt)
    
    # build plots for each trial
    for i, trial in enumerate(trials["_trials"]):
        out_txt_trial = f"trial: {i}"

        # obtain results from the trial
        result = trial["result"]

        # extract hyper parameters from the results
        h_parm = result["hyper_parameters"]
        title_plot = f"Training Results - Job: {job_id}"
        if len(trials["_trials"]) > 1:
            title_plot += f" - Trial: {i}"
        for key in h_parm:
            out_txt_trial += "\n  {:12}\t  {}".format(key, h_parm[key])

        # add final loss and cost to the info
        out_txt_trial += f"\n{'with loss:':18}{result['loss']}"
        out_txt_trial += f"\n{'with final cost:':18}{result['final_cost']}"
                
        # generate the plots
        cost_condition_plot(result, title_plot, out_file=out_dir+ "/"+ f"trial_{i}")
        if ("roc_auc" in result["cost_data"]) and (result["cost_data"]["roc_auc"]):
            cost_auc_plot(result, title_plot, out_file=out_dir+ "/cost_auc_"+ f"trial_{i}")
        out_txt_trial += "\n\n"
        print(out_txt_trial)
        out_txt += out_txt_trial
        
    # save info on all trials
    txt_file = open(out_dir + "/info.txt", "w")
    txt_file.write(out_txt)
    txt_file.close()
    print("")

