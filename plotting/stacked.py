import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import os


def stacked_plot_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
    )
    plt.xlabel(feature)
    plt.ylabel("N")
    plt.legend()


def stacked_plot_normalised_to_max_first_entries(
    anomaly, normal, feature, jet_info=None
):
    norm_normal = ak.firsts(normal[feature]) / np.max(ak.firsts(normal[feature]))
    norm_anomaly = ak.firsts(anomaly[feature]) / np.max(ak.firsts(anomaly[feature]))

    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [norm_normal, norm_anomaly],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [
            ak.firsts(normal[feature])[: len(ak.firsts(anomaly[feature]))],
            ak.firsts(anomaly[feature]),
        ],
        stacked=True,
        label=[f"normal", "anomalous"],
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_normalised_first_entries(
    anomaly, normal, feature, jet_info=None
):
    plt.figure(
        f"Distribution histogram normalised anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [
            ak.firsts(normal[feature])[: len(ak.firsts(anomaly[feature]))],
            ak.firsts(anomaly[feature]),
        ],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_normalised_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {feature}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(feature)
    plt.ylabel("sigma")
    plt.legend()


# quark vs gluon stacked plots
def stacked_plots_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.title(title)
        plt.hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            label=["g_normal", "g_anomalous", "q_normal", "q_anomalous"],
            bins=50
        )
        plt.xlabel(feature)
        plt.ylabel("N")
        plt.legend()
        # save plot
        plt.savefig(out_dir + "/first_entries_qg_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_last_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.title(title)
        plt.hist(
            [[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))],
            [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))],
            [q_normal[feature][i][-1] for i in range(len(q_normal[feature]))],
            [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]],

            stacked=True,
            label=["g_normal", "g_anomalous", "q_normal", "q_anomalous"],
            bins=50
        )
        plt.xlabel(feature)
        plt.ylabel("N")
        plt.legend()
        # save plot
        plt.savefig(out_dir + "/last_entries_qg_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_normalised_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.title(title)
        plt.hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            density=True,
            label=["g_normal", "g_anomalous", "q_normal", "q_anomalous"],
            bins=50
        )
        plt.xlabel(feature)
        plt.ylabel("N")
        plt.legend()
        # save plot
        plt.savefig(out_dir + "/normalised_first_entries_qg_" + feature)
        plt.close()  # close figure - clean memory


def stacked_plots_all_splits_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}, all splittings"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.title(title)
        plt.hist(
            [ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature]), ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])],
            stacked=True,
            label=["g_normal", "g_anomalous", "q_normal", "q_anomalous"],
            bins=50
        )
        plt.xlabel(feature)
        plt.ylabel("N")
        plt.legend()
        # save plot
        plt.savefig(out_dir + "/all_splits_qg_" + feature)
        plt.close()  # close figure - clean memory
        

# side - by - side
def stacked_plots_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 8],
        )
        fig.suptitle(title)
        
        ax[0].hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])],
            stacked=True,
            label=["g_normal", "g_anomalous"],
            bins=50
        )
        ax[0].set_xlabel(feature)
        ax[0].set_ylabel("N")
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            label=["q_normal", "q_anomalous"],
            bins=50
        )
        ax[1].set_xlabel(feature)
        ax[1].set_ylabel("N")
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/first_entries_qg_sided_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_last_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 8],
        )
        fig.suptitle(title)
        
        ax[0].hist(
            [[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))], [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))]],
            stacked=True,
            label=["g_normal", "g_anomalous"],
            bins=50
        )
        ax[0].set_xlabel(feature)
        ax[0].set_ylabel("N")
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [[q_normal[feature][i][-1] for i in range(len(q_normal[feature]))], [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]],
            stacked=True,
            label=["q_normal", "q_anomalous"],
            bins=50
        )
        ax[1].set_xlabel(feature)
        ax[1].set_ylabel("N")
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/last_entries_qg_sided_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_normalised_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 8],
        )
        fig.suptitle(title)
        
        ax[0].hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])],
            stacked=True,
            density=True,
            label=["g_normal", "g_anomalous"],
            bins=50
        )
        ax[0].set_xlabel(feature)
        ax[0].set_ylabel("N")
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            density=True,
            label=["q_normal", "q_anomalous"],
            bins=50
        )
        ax[1].set_xlabel(feature)
        ax[1].set_ylabel("N")
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/normalised_first_entries_qg_sided_" + feature)
        plt.close()  # close figure - clean memory
        
def stacked_plots_all_splits_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution histogram anomalies quarks and gluons for {feature}"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 8],
        )
        fig.suptitle(title)
        
        ax[0].hist(
            [ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature])],
            stacked=True,
            label=["g_normal", "g_anomalous"],
            bins=50
        )
        ax[0].set_xlabel(feature)
        ax[0].set_ylabel("N")
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])],
            stacked=True,
            label=["q_normal", "q_anomalous"],
            bins=50
        )
        ax[1].set_xlabel(feature)
        ax[1].set_ylabel("N")
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/all_splits_qg_sided_" + feature)
        plt.close()  # close figure - clean memory