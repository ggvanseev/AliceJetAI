"""
Contains all the cost condition and cost functions used in plotting.
"""

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pylab as plt
import os
import seaborn as sns

parameter_names ={
    "batch_size" : "Batch Size",
    "dropout" : "Dropout",
    "hidden_dim" : "Hidden Dimensions",
    "learning_rate" : "Learning Rate",
    "min_epochs" : "Minimum Epochs",
    "num_layers" : "Number of Layers",
    "output_dim" : "Output Dimension",
    "pooling" : "Pooling Type",
    "scaler_id" : "Scaler Type",
    "svm_gamma" : r"OCSVM $\gamma$",
    "svm_nu" : r"OCSVM $\nu$",
    "variables" : "Variables Used",
    "loss": "Loss",
    "final_cost": "Cost",
}

def violin_plots(df, min_val, min_df, parameters, job_ids, test_param="loss", yscale="linear"):
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
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 16})
    
    # store violin plots in designated directory
    out_dir = f"output/violin_plots_"+yscale
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
            fig, ax = plt.subplots(figsize=(7 * 1.618, 7))
            ax2 = sns.violinplot(x=p_name, y=test_param, cut=0, data=df)

            # plot minimum loss per parameter value
            unique = sorted(df[p_name].unique())
            for p_val in min_df[p_name].unique():
                label = "Minimum:\n{} = {}\n{} = {:.2E}".format(
                    parameter_names[parameter], p_val, parameter_names[test_param], min_val
                )
                ax2.scatter(int([lab.get_text() for lab in ax2.get_xticklabels()].index(str(p_val))), min_val, s=120, marker="x", c="r", linewidth=2, zorder=3, label=label)

            # rotate x-ticks
            if parameter != "variables":
                _ = plt.xticks(rotation=45, ha="right")
            else:
                a = 1
                _ = plt.xticks(rotation=9, ha="right")
                

            # create appropriate title and x-label
            plt.xlabel(parameter_names[parameter])
            plt.ylabel("Loss" if test_param == "loss" else "Final Cost")
            plt.yscale(yscale) 
            plt.legend()
            plt.tight_layout()
            
            out_file = out_dir + "/violin_plot_" + test_param + "_vs_" + parameter
            
            # save version without title
            fig.savefig(out_file+"_no_title")
            plt.title(
                parameter.replace("_", " ").title()
                + " vs "
                + test_param.replace("_", " ").title()
                + " - job(s): "
                + ",".join(job_ids)
            ) # save title afterwards
            plt.tight_layout()

            # save plot
            plt.savefig(out_file)
            plt.close('all')  # close figure - clean memory
        print(f"\n Violin Plots Stored in {out_dir}")
