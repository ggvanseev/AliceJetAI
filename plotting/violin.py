"""
Contains all the cost condition and cost functions used in plotting.
"""

import matplotlib.pyplot as plt
import os
import seaborn as sns

def violin_plots(df, min_val, min_df, parameters, job_ids, test_param="loss"):
    """Function that creates and stores violin plots. The function accepts
    a Pandas dataframe of all the trials of a training. The violin plots
    showcase the density of the loss for all trained hyperparameters as well
    as the minimum loss. Each trial contains one LSTM and one OC-SVM model.

    Args:
        df (pandas.DataFrame): Dataframe storing all trials results
        min_val (float): Minimum loss value
        min_df (pandas.DataFrame): Dataframe storing trial(s) with the minimum loss
        parameters (dict_keys): List of hyperparameter names
        job_ids (str): Ids given to this training/run
        test_param (str, optional): Parameter which was tested. Can be loss or cost 
                                    or something else. Defaults to "loss".
    """    
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
            # plt.yscale("log") TODO
            plt.legend()

            # save plot
            plt.savefig(out_dir + "/violin_plot_" + test_param + "_vs_" + parameter)
            plt.close(fig) # close figure - clean memory
