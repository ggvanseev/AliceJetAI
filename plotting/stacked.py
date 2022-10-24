import matplotlib.pyplot as plt
# set font size

import awkward as ak
import numpy as np
import os

from branch_names import variable_names as vn

def stacked_plot(data, title, x_label, out_file, labels=["Gluon - Normal", "Gluon - Anomaly", "Quark - Normal", "Quark - Anomaly"]):
    """Make a general stacked plot of given qg data

    Args:
        data (_type_): _description_
        title (_type_): _description_
        bins (_type_): _description_
        feature (_type_): _description_
        out_dir (_type_): _description_
    """
    
    # set matplotlib font size
    plt.rcParams.update({'font.size': 24})
    
    # set bin sizes
    binwidth = (ak.max(data) - ak.min(data)) / 50
    bins = np.arange(ak.min(data), ak.max(data) + binwidth, binwidth)
    
    # plot histograms
    plt.figure(
        title,
        figsize=[1.36 * 8, 8],
    )
    plt.hist(
        data,
        stacked=True,
        label=labels,
        bins=bins,
        alpha=0.8,
        zorder=3,
    )
    
    # plot setup
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.grid(alpha=0.4)
    plt.legend()
    
    # save plot
    plt.savefig(out_file + "_no_title")
    plt.title(title)
    plt.savefig(out_file)
    plt.close('all')  # close figure - clean memory
    
    return


def stacked_plot_sided(data, fig_title, x_label, out_file, titles=["Gluon Jets", "Quark Jets"], labels=["Normal", "Anomaly"]):
    """Make a general side by side stacked histogram of qg data

    Args:
        data (_type_): _description_
        title (_type_): _description_
        bins (_type_): _description_
        feature (_type_): _description_
        out_dir (_type_): _description_
    """
    
    # set matplotlib font size
    plt.rcParams.update({'font.size': 24})
    
    # set binwidth
    binwidth = (ak.max(data) - ak.min(data)) / 50
    bins = np.arange(ak.min(data), ak.max(data) + binwidth, binwidth)
        
    # plot histograms
    fig, ax = plt.subplots(1, 2,
        sharex=True,
        sharey=True,
        figsize=[12, 7],
    )
    for i, title in enumerate(titles):
        hist = ax[i].hist(
            data[i],
            stacked=True,
            label=labels,
            bins=bins,
            alpha=0.8,
            zorder=3,
        )
        # fraction anomalies
        anom_ax = ax[i].twinx()
        anom_ax.set_ylim(bottom=0, top=1)
        if i == 0:
            anom_ax.set_yticklabels([])
        anom_fraction = (hist[0][1] - hist[0][0]) /hist[0][1] # anomalies ( = (total - normal) ) / total
        anom_plot = anom_ax.errorbar((hist[1][:-1] + hist[1][1:]) / 2,
                                     anom_fraction.tolist(), 
                                     xerr=binwidth/2, 
                                     color='r', 
                                     linestyle='', 
                                     elinewidth=2, 
                                     zorder=3, 
                                     label= "Anomalous / Total")
        anom_plot[0].set_clip_on(False) # so the plot can overlap the axis

        # plot setup
        ax[i].set_title(title)
        ax[i].set_xlabel(x_label)
        ax[i].grid(alpha=0.4)
    # legend & overall setup
    ax[0].set_ylabel("Count")
    anom_ax.set_ylabel(r"Fraction")
    legend = [h[0] for h in hist[2]] + [anom_plot]
    labels = [l.get_label() for l in legend]
        
    fig.subplots_adjust(left=0.1, right=1., top=0.85, bottom=0.12, hspace=0.3, wspace=0.1)
    lgd = fig.legend(legend, labels, loc='upper center', bbox_to_anchor=(0.55, 0), 
                     fancybox=True, shadow=True, ncol=3) # , prop={'size': 8}
        
    # save plot
    plt.savefig(out_file +"_no_title", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=160)
    plt.title(fig_title)
    plt.savefig(out_file, bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=160)
    plt.close('all')  # close figure - clean memory
    
    return


def stacked_plot_sided_old(data, fig_title, x_label, out_file, titles=["Gluon Jets", "Quark Jets"], labels=["Normal", "Anomaly"]):
    """Make a general side by side stacked histogram of qg data.
    Old version, does not contain the anomal / total fraction plot.
    This is actually the better version for now, because the other 
    one is too messy.

    Args:
        data (_type_): _description_
        title (_type_): _description_
        bins (_type_): _description_
        feature (_type_): _description_
        out_dir (_type_): _description_
    """
    
    # set matplotlib font size
    plt.rcParams.update({'font.size': 24})
    
    # set binwidth
    binwidth = (ak.max(data) - ak.min(data)) / 50
    bins = np.arange(ak.min(data), ak.max(data) + binwidth, binwidth)
        
    # plot histograms
    fig, ax = plt.subplots(1, 2,
        sharex=True,
        sharey=True,
        figsize=[12, 7],
    )
    for i, title in enumerate(titles):
        ax[i].hist(
            data[i],
            stacked=True,
            label=labels,
            bins=bins,
            alpha=0.8,
            zorder=3,
        )
        
        # plot setup
        ax[i].set_title(title)
        ax[i].set_xlabel(x_label)
        ax[i].grid(alpha=0.4)
    
    # fig setup & save file
    ax[0].set_ylabel("Count")
    ax[1].legend()
    fig.subplots_adjust(bottom=0.15, wspace=0.1)
    plt.savefig(out_file + "_no_title")
    plt.title(fig_title)
    plt.savefig(out_file)
    plt.close('all')  # close figure - clean memory
    
    return


# quark vs gluon stacked plots
def stacked_plots_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), 
                    ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histogram Anomalies Quarks And Gluons For {vn[feature]} - First Splittings"
        label = vn[feature]
        out_file = out_dir + f"/trial{trial}_first_" + feature 
        stacked_plot(data, title, label, out_file)


def stacked_plots_last_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))],
                [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))],
                [q_normal[feature][i][-1] for i in range(len(q_normal[feature]))],
                [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Last Splittings"
        label = vn[feature]
        out_file = out_dir + f"/trial{trial}_last_" + feature 
        stacked_plot(data, title,feature, out_file)


def stacked_plots_normalised_first_entries_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    for feature in features:
        try:
            data = [ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature]), 
                    ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Normalised First Splittings"
        label = "Normalised First Splitting " + vn[feature]
        out_file = out_dir + f"/trial{trial}_normalised_first_" + feature 
        stacked_plot(data, title, label, out_file)


def stacked_plots_mean_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[ak.mean(x) for x in g_normal[feature]], [ak.mean(x) for x in g_anomaly[feature]], 
                [ak.mean(x) for x in q_normal[feature]], [ak.mean(x) for x in q_anomaly[feature]]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Mean Of Jets"
        label = "Mean " + vn[feature]
        out_file = out_dir + f"/trial{trial}_mean_" + feature 
        stacked_plot(data, title, label, out_file)
    
    
def stacked_plots_splittings_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        data = [[ak.count(x) for x in g_normal[feature]], [ak.count(x) for x in g_anomaly[feature]], 
                [ak.count(x) for x in q_normal[feature]], [ak.count(x) for x in q_anomaly[feature]]]
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Splittings Of Jet"
        label = "Nr. Of Splittings " + vn[feature]
        out_file = out_dir + f"/trial{trial}_splittings_" + feature 
        stacked_plot(data, title, label, out_file)


def stacked_plots_all_splits_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature]), 
                    ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - All Splittings"
        label = "All Splittings " + vn[feature]
        out_file = out_dir + f"/trial{trial}_all_" + feature 
        stacked_plot(data, title, label, out_file)
        

# side - by - side
def stacked_plots_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])], 
                    [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - First Splittings"
        label = "First Splitting " + vn[feature]
        out_file = out_dir + f"/trial{trial}_first_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_first_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)


def stacked_plots_last_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))], [g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))]], 
                    [[q_normal[feature][i][-1] for i in range(len(q_normal[feature]))], [q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))]]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Last Splittings"
        label = "Last Splitting " + vn[feature]
        out_file = out_dir + f"/trial{trial}_last_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_last_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)
        
    
def stacked_plots_mean_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[[ak.mean(x) for x in g_normal[feature]], [ak.mean(x) for x in g_anomaly[feature]]],
                    [[ak.mean(x) for x in q_normal[feature]], [ak.mean(x) for x in q_anomaly[feature]]]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks and Gluons for {vn[feature]} - Mean Of Jets"
        label = "Mean " + vn[feature]
        out_file = out_dir + f"/trial{trial}_mean_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_mean_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)


def stacked_plots_splittings_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[[ak.count(x) for x in g_normal[feature]], [ak.count(x) for x in g_anomaly[feature]]],
                    [[ak.count(x) for x in q_normal[feature]], [ak.count(x) for x in q_anomaly[feature]]]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Splittings Of Jets"
        label = "Nr. Of Splittings " + vn[feature]
        out_file = out_dir + f"/trial{trial}_splittings_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_splittings_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)


def stacked_plots_normalised_first_entries_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[ak.firsts(g_normal[feature]), ak.firsts(g_anomaly[feature])],
                    [ak.firsts(q_normal[feature]), ak.firsts(q_anomaly[feature])]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {vn[feature]} - Normalised First Splittings"
        label = "Normalised First Splitting " + vn[feature]
        out_file = out_dir + f"/trial{trial}_normalised_first_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_normalised_first_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)
  
        
def stacked_plots_all_splits_qg_sided(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, trial):
    # store stacked plots in designated directory
    out_dir = f"output/stacked_plots_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
        
    for feature in features:
        try:
            data = [[ak.flatten(g_normal[feature]), ak.flatten(g_anomaly[feature])],
                    [ak.flatten(q_normal[feature]), ak.flatten(q_anomaly[feature])]]
        except ValueError:
            print(f"Either no normal or no anomalous data for {job_id} trial {trial}!")
            return -1
        label = "All Splittings " + vn[feature]
        title = f"Distribution Histograms Anomalies Quarks And Gluons For {label} - All Splittings"
        out_file = out_dir + f"/trial{trial}_all_sided_" + feature 
        # stacked_plot_sided(data, title, label, out_file)
        # out_file = out_dir + f"/zold_trial{trial}_all_sided_" + feature 
        stacked_plot_sided_old(data, title, label, out_file)
