import torch
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "9756505",
]

# select test parameter: e.g. "loss" or "final_cost"
test_param = "loss"

# store violin plots in designated directory
out_dir = f"output/violin_plots"
for job_id in job_ids:
    out_dir += f"_{job_id}"
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

# load trials results from file and
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trials_test_list = [
    torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# reform to complete list of trials
trials_list = [trial for trials in [trials["_trials"] for trials in trials_test_list] for trial in trials]

# build DataFrame
df = pd.concat([pd.json_normalize(trial["result"]) for trial in trials_list])
df = df[df["loss"] != 10]  # filter out bad model results

# get minima
min_val = df[test_param].min()
min_df = df[df[test_param] == min_val].reset_index()

# print best model(s) hyperparameters:
print("\nBest Hyper Parameters:")
hyper_parameters_df = min_df.loc[:, min_df.columns.str.startswith('hyper_parameters')]
for index, row in hyper_parameters_df.iterrows():
    print(f"\nModel {index}:")
    for key in hyper_parameters_df.keys():
        print("  {:10}\t  {}".format(key.split('.')[1], row[key]))
    print(f"with loss: \t\t{min_df['loss'].iloc[index]}") 
    print(f"with final cost:\t{min_df['final_cost'].iloc[index]}")   

# loop over each hyperparameter used in the analysis
parameters = trials_list[0]["result"]["hyper_parameters"].keys()
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
