import matplotlib.pyplot as plt
import time
import os
import seaborn as sns


def plot_cost_vs_cost_condition(
    track_cost, track_cost_condition, title_plot, show_flag=False, save_flag=False
):

    fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
    fig.suptitle(title_plot, y=1.08)
    ax1.plot(track_cost_condition[1:])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cost Condition")

    ax2 = ax1.twinx()
    ax2.plot(track_cost[1:], "--", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Cost")
    plt.legend()

    if save_flag:
        fig.savefig("output/" + title_plot + str(time.time()) + ".png")
    if show_flag:
        plt.show()


def cost_condition_plot(result, title_plot):
    # extract cost data from the results
    cost_data = result["cost_data"]
    if cost_data != 10:  # check for failed models
        track_cost = cost_data["cost"]
        track_cost_condition = cost_data["cost_condition"]
    else:
        out_txt += "\n *** FAILED MODEL *** \n"
        return -1

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
    return fig


def cost_condition_plots(pickling_trials, job_id):

    # make out directory if it does not exist yet
    out_dir = f"output/cost_condition_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # store trial parameters
    out_txt = ""

    # build plots for each trial
    for i, trial in enumerate(pickling_trials["_trials"]):
        out_txt += f"trial: {i}"

        # obtain results from the trial
        result = trial["result"]

        # extract hyper parameters from the results
        h_parm = result["hyper_parameters"]
        title_plot = f""
        for key in h_parm:
            title_plot += f"{h_parm[key]}_{key}_"
            out_txt += "\n  {:12}\t  {}".format(key, h_parm[key])

        # generate the plot
        fig = cost_condition_plot(result, title_plot)
        if fig == -1:
            break

        # save and close the plot
        fig.savefig(out_dir + "/" f"trial_{i}.png")
        plt.close(fig)  # close figure - clean memory

        out_txt += "\n\n"

    # save info on all trials
    txt_file = open(out_dir + "/info.txt", "w")
    txt_file.write(out_txt)
    txt_file.close()
    return


def violin_plots(df, min_val, min_df, parameters, job_ids, test_param="loss"):
    # store violin plots in designated directory
    out_dir = f"output/violin_plots"
    for job_id in job_ids:
        out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    # loop over each hyperparameter used in the analysis
    if df.empty:
        print("No good models obtained from the results")
    else:
        for parameter in parameters:
            # full parameter name
            p_name = df.keys()[[parameter in key for key in df.keys()]][0]

            # plot violin plot
            fig, ax = plt.subplots(figsize=(9, 6))
            ax2 = sns.violinplot(x=p_name, y=test_param, cut=0, data=df)

            # plot minimum loss per parameter value
            unique = sorted(df[p_name].unique())
            for p_val in min_df[p_name].unique():
                label = "Minimum:\n{} = {}\n{} = {:.2E}".format(
                    parameter, p_val, test_param, min_val
                )
                ax2.plot(int(unique.index(p_val)), min_val, "o", label=label)

            # rotate x-ticks
            _ = plt.xticks(rotation=45, ha="right")

            # create appropriate title and x-label
            plt.title(
                parameter.replace("_", " ").title()
                + " vs "
                + test_param.replace("_", " ").title()
                + " - job(s): "
                + ",".join(job_ids)
            )
            plt.xlabel(parameter)
            plt.yscale("log")
            plt.legend()

            # save plot
            plt.savefig(out_dir + "/violin_plot_" + test_param + "_vs_" + parameter)
            plt.close('all')
