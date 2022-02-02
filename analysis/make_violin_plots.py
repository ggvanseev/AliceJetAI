import pickle
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load trials results from file and convert to pandas dataframe
trials = pickle.load(open("storing_results/trials_test.p", "rb"))
df = pd.json_normalize(trials.results)

# loop over each hyperparameter used in the analysis
parameters = trials.results[0]["hyper_parameters"].keys()
for parameter in parameters:

    # full parameter name
    p_name = df.keys()[[parameter in key for key in df.keys()]][0]

    # plot violin plot
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(x=p_name, y="loss", data=df)

    # plot minimum loss per parameter value
    minimum = df[["loss", p_name]].groupby(p_name).min().reset_index()
    sns.swarmplot(x=p_name, y="loss", data=minimum, color="r")

    # rotate x-ticks
    _ = plt.xticks(rotation=45, ha="right")

    plt.title(parameter.replace("_", " ").title())
    plt.xlabel(parameter)
    plt.savefig("output/violin_plot_" + parameter)
