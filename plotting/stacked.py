import matplotlib.pyplot as plt
# set font size
plt.rcParams.update({'font.size': 13.5})
import awkward as ak
import numpy as np
import os

from branch_names import variable_names as vn

def stacked_plot_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {vn[feature]}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
    )
    plt.xlabel(vn[feature])
    plt.ylabel("N")
    plt.legend()


def stacked_plot_normalised_to_max_first_entries(
    anomaly, normal, feature, jet_info=None
):
    norm_normal = ak.firsts(normal[feature]) / np.max(ak.firsts(normal[feature]))
    norm_anomaly = ak.firsts(anomaly[feature]) / np.max(ak.firsts(anomaly[feature]))

    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {vn[feature]}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [norm_normal, norm_anomaly],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(vn[feature])
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram anomalies {jet_info} for {vn[feature]}",
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
    plt.xlabel(vn[feature])
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_same_n_entries_normalised_first_entries(
    anomaly, normal, feature, jet_info=None
):
    plt.figure(
        f"Distribution histogram normalised anomalies {jet_info} for {vn[feature]}",
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
    plt.xlabel(vn[feature])
    plt.ylabel("sigma")
    plt.legend()


def stacked_plot_normalised_first_entries(anomaly, normal, feature, jet_info=None):
    plt.figure(
        f"Distribution histogram normalised to max anomalies {jet_info} for {vn[feature]}",
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        [ak.firsts(normal[feature]), ak.firsts(anomaly[feature])],
        stacked=True,
        label=[f"normal", "anomalous"],
        density=True,
    )
    plt.xlabel(vn[feature])
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
        title = f"Distribution Histogram Anomalies Quarks And Gluons For {vn[feature]} - First Splittings"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot
        plt.savefig(out_dir + "/first_entries_qg_" + feature+"_no_title")
        plt.title(title)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Last Splittings"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.hist(
            [[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))],
            [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))],
            [q_normal[feature][i][-1] for i in range(len(q_normal[feature]))],
            [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]],

            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot
        plt.savefig(out_dir + "/last_entries_qg_" + feature+"_no_title")
        plt.title(title)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Normalised First Splittings"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            density=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot
        plt.savefig(out_dir + "/normalised_first_entries_qg_" + feature + "_no_title")
        plt.title(title)
        plt.savefig(out_dir + "/normalised_first_entries_qg_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_mean_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Mean Of Jets"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.hist(
            [[ak.mean(x) for x in g_normal[feature]], [ak.mean(x) for x in g_anomaly[feature]], [ak.mean(x) for x in q_normal[feature]], [ak.mean(x) for x in q_anomaly[feature]]],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot
        plt.savefig(out_dir + "/mean_qg_" + feature+"_no_title")
        plt.title(title)
        plt.savefig(out_dir + "/mean_qg_" + feature)
        plt.close()  # close figure - clean memory
    
def stacked_plots_splittings_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Splittings Of Jet"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.hist(
            [[ak.count(x) for x in g_normal[feature]], [ak.count(x) for x in g_anomaly[feature]], [ak.count(x) for x in q_normal[feature]], [ak.count(x) for x in q_anomaly[feature]]],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            histtype = 'stepfilled',
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot        
        plt.savefig(out_dir + "/splittings_qg_" + feature +"_no_title")
        plt.title(title)
        plt.savefig(out_dir + "/splittings_qg_" + feature)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - All Splittings"
        plt.figure(
            title,
            figsize=[1.36 * 8, 8],
        )
        plt.title(title)
        plt.hist(
            [ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature]), ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        plt.xlabel(vn[feature])
        plt.ylabel("N")
        plt.grid(alpha=0.4)
        plt.legend()
        
        # save plot
        plt.savefig(out_dir + "/all_splits_qg_" + feature+ "_no_title")        
        plt.title(title)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - First Splittings"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        fig.suptitle(title)
        
        ax[0].hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3,
        )
        ax[0].set_xlabel(vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            label=["Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        ax[1].set_xlabel(vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/first_entries_qg_sided_" + feature +"_no_title")
        plt.title(title)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Last Splittings"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        
        ax[0].hist(
            [[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))], [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))]],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        ax[0].set_xlabel("Last Splitting" + vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [[q_normal[feature][i][-1] for i in range(len(q_normal[feature]))], [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]],
            stacked=True,
            label=["Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        ax[1].set_xlabel("Last Splitting " + vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/last_entries_qg_sided_" + feature + "_no_title")
        fig.suptitle(title)
        plt.savefig(out_dir + "/last_entries_qg_sided_" + feature)
        plt.close()  # close figure - clean memory
        
    
def stacked_plots_mean_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution Histograms Anomalies Quarks and Gluons for {vn[feature]} - Mean Of Jets"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        
        ax[0].hist(
            [[ak.mean(x) for x in g_normal[feature]], [ak.mean(x) for x in g_anomaly[feature]]],
            stacked=True,
            density=True,
            alpha=0.8,
            label=["Normal", "Anomaly"],
            bins=50,            
            zorder=3,
        )
        ax[0].set_title("Gluons")
        ax[0].set_xlabel("Mean " + vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [[ak.mean(x) for x in q_normal[feature]], [ak.mean(x) for x in q_anomaly[feature]]],
            stacked=True,
            density=True,
            alpha=0.8,
            label=["Normal", "Anomaly"],
            bins=50,
            zorder=3,
        )
        ax[1].set_title("Quarks")
        ax[1].set_xlabel("Mean " + vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/mean_qg_sided_" + feature + "_no_title")
        fig.suptitle(title)
        plt.savefig(out_dir + "/mean_qg_sided_" + feature)
        plt.close()  # close figure - clean memory

def stacked_plots_splittings_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Splittings Of Jets"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        
        g = [[ak.count(x) for x in g_normal[feature]], [ak.count(x) for x in g_anomaly[feature]]]
        ax[0].hist(
            g,
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly"],
            bins = ak.max(g)
        )
        ax[0].set_xlabel(vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")

        q = [[ak.count(x) for x in q_normal[feature]], [ak.count(x) for x in q_anomaly[feature]]]
        ax[1].hist(
            q,
            stacked=True,
            label=["Quark - Normal", "Quark - Anomaly"],
            bins = ak.max(q) 
        )
        ax[1].set_xlabel(vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/splittings_qg_sided_" + feature + "_no_title")
        fig.suptitle(title)
        plt.savefig(out_dir + "/splittings_qg_sided_" + feature)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Normalised First Splittings"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        
        ax[0].hist(
            [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])],
            stacked=True,
            density=True,
            label=["Gluon - Normal", "Gluon - Anomaly"],
            bins=50
        )
        ax[0].set_xlabel(vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])],
            stacked=True,
            density=True,
            label=["Quark - Normal", "Quark - Anomaly"],
            bins=50
        )
        ax[1].set_xlabel(vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/normalised_first_entries_qg_sided_" + feature+ "_no_title")
        fig.suptitle(title)
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
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - All Splittings"
        fig, ax = plt.subplots(1, 2,
            sharex=True,
            sharey=True,
            figsize=[1.36 * 8, 6],
        )
        
        ax[0].hist(
            [ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature])],
            stacked=True,
            label=["Gluon - Normal", "Gluon - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        ax[0].set_xlabel(vn[feature])
        ax[0].set_ylabel("N")
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc="upper right")
        
        ax[1].hist(
            [ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])],
            stacked=True,
            label=["Quark - Normal", "Quark - Anomaly"],
            bins=50,
            alpha=0.8,
            zorder=3
        )
        ax[1].set_xlabel(vn[feature])
        ax[1].set_ylabel("N")
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc="upper right")
        
        # save plot
        plt.savefig(out_dir + "/all_splits_qg_sided_" + feature + "_no_title")
        fig.suptitle(title)
        plt.savefig(out_dir + "/all_splits_qg_sided_" + feature)
        plt.close()  # close figure - clean memory