import torch
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# select file monickers to be analysed e.g. ../trials_test_{monicker}.p
job_ids = [
    "9639018.burrell.nikhef.nl",
]

# select test parameter: e.g. loss or cost
test_param = "final_cost"

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
trials_list = [
    torch.load(f"storing_results/trials_test_{job_id}.p", map_location=device)
    for job_id in job_ids
]

# build DataFrame
df = pd.concat([pd.json_normalize(trial.results) for trial in trials_list])
df = df[df["loss"] != 10]  # filter out bad model results
min_val = df[test_param].min()
min_df = df[df[test_param] == min_val].reset_index()

# loop over each hyperparameter used in the analysis
parameters = trials_list[0].results[0]["hyper_parameters"].keys()
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
        minimum = df[df.loss == df.loss.min()][[test_param, p_name]]
        label = "Minimum:\n{} = {}\n{} = {:.2E}".format(
            parameter, min_df[p_name][0], test_param, min_val
        )
        # minimum = df[[test_param, p_name]].groupby(p_name).min().reset_index()
        # sns.swarmplot(x=p_name, y=test_param, data=minimum, color="r", label=label)

        unique = sorted(df[p_name].unique())
        ax2.plot(int(unique.index(min_df[p_name][0])), min_val, "ro", label=label)

        # rotate x-ticks
        _ = plt.xticks(rotation=45, ha="right")

        # create appropriate title and x-label
        plt.title(
            parameter.replace("_", " ").title()
            + " vs "
            + test_param.replace("_", " ").title()
            + " - jobs: "
            + ",".join(job_ids)
        )
        plt.xlabel(parameter)
        plt.legend()

        # save plot
        plt.savefig(out_dir + "/violin_plot_" + test_param + "_vs_" + parameter)
