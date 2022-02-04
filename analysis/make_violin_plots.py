import pickle
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# select test parameter: loss or cost
test_param = "loss"

# load trials results from file and
trials = pickle.load(open("storing_results/trials_test.p", "rb"))

# build DataFrame
df = pd.json_normalize(trials.results)
df = df[df["loss"] != 10]  # filter out bad model results

# loop over each hyperparameter used in the analysis
parameters = trials.results[0]["hyper_parameters"].keys()
if df.empty:
    print("No good models obtained from the results")
else:
    for parameter in parameters:

        # full parameter name
        p_name = df.keys()[[parameter in key for key in df.keys()]][0]

        # plot violin plot
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(x=p_name, y=test_param, data=df)

        # plot minimum loss per parameter value
        minimum = df[[test_param, p_name]].groupby(p_name).min().reset_index()
        sns.swarmplot(x=p_name, y=test_param, data=minimum, color="r")

        # rotate x-ticks
        _ = plt.xticks(rotation=45, ha="right")

        # create appropriate title and x-label
        plt.title(parameter.replace("_", " ").title())
        plt.xlabel(parameter)

        # save plot
        plt.savefig("output/violin_plot_" + test_param + "_vs_" + parameter)
