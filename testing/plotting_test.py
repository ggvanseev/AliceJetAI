from attr import s
import matplotlib.pyplot as plt
import awkward as ak
from matplotlib import colors
import random
import numpy as np
import os

import branch_names as na
from functions.run_lstm import calc_lstm_results
from functions.data_manipulation import (
    lstm_data_prep,
    h_bar_list_to_numpy,
    single_branch,
)
from plotting.roc import ROC_plot_curve

def sample_plot(data, idx=None, label=""):
    # random sample if no index is given
    if idx == None:
        idx = int(random.random() * len(data))
        
    sample = data[idx]
    x = [i[0] for i in sample]
    y = [i[1] for i in sample]
    plt.scatter(x, y, label=label)
    plt.xlim(np.min(sample),np.max(sample))
    plt.ylim(np.min(sample),np.max(sample))


def create_sample_plot(data, idx=None):
    plt.figure()
    sample_plot(data, idx)
    plt.show()
    return
    

def normal_vs_anomaly_2D(data, classification, file_name):
     
    normal = data[classification == 1]
    anomalous = data[classification == -1]

    plt.figure(figsize=[6 * 1.36, 10 * 1.36], dpi=160)

    # plot normal
    x = [i[0] for i in normal]
    y = [i[1] for i in normal]
    plt.scatter(x, y, color='blue', label='normal')

    # plot anomalous
    x = [i[0] for i in anomalous]
    y = [i[1] for i in anomalous]
    plt.scatter(x, y, color='red', label='anomalous')
    
    plt.legend()
    #plt.show()
    plt.savefig(f"testing/output/{file_name}")
    plt.close('all')


def normal_vs_anomaly_2D_all(data_dict, classification_dict, ocsvm_list, file_name, job_id=None, y=None, xlabel=r"$\overline{h_{i,1}}$", ylabel=r"$\overline{h_{i,2}}$"):
    
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 18})
    
    first_key = list(data_dict.keys())[0]
    nr_trials = len(data_dict[first_key])
    nr_digits = len(data_dict.keys())
    
    markers = ["o", "s", "v", "*" , "D" , "-"]
    
    for i in range(nr_trials):
        
        fig = plt.figure(figsize=[6 * 1.36, 6], dpi=160)
        ax = plt.subplot(111)
        model = ocsvm_list[i]
        
        # get min / max this trial
        xmin = min([min(data_dict[key][i][:,0]) for key in data_dict])
        xmax = max([max(data_dict[key][i][:,0]) for key in data_dict])
        ymin = min([min(data_dict[key][i][:,1]) for key in data_dict])
        ymax = max([max(data_dict[key][i][:,1]) for key in data_dict])
        
        margin_x = 0.05 * (xmax-xmin)
        margin_y = 0.05 * (ymax-ymin)
            
        # meshgrid for plots
        xx1, yy1 = np.meshgrid(np.linspace(xmin - margin_x, xmax + margin_x, 500),
                            np.linspace(ymin - margin_y, ymax + margin_y, 500))
        
        # decision function
        # Z1 = model.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        # Z1 = Z1.reshape(xx1.shape)
        Z1 = model.predict(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        Z1 = -Z1
        
        # plot data and decision function
        hyperplane = ax.contour(xx1, yy1, Z1, levels=(0), linewidths=(1.0),
                    linestyles=('-'), colors=['r','k','b'])
        ax.contourf(xx1, yy1, Z1, cmap=plt.cm.coolwarm, alpha=0.08, linestyles="None")
        ax.clabel(hyperplane, hyperplane.levels, inline=True, fmt="  Hyperplane  ", manual=[(xmin+2*margin_x,ymin+4*margin_y)], fontsize=8)
        
        # plot support vectors
        # plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c=y[model.support_] if y else model.predict(model.support_vectors_),
        #        cmap=plt.cm.viridis, lw=1, edgecolors='k', label="Support Vectors")
        
        color = ['b', 'tab:orange', 'c', 'm', 'k', 'b1', 'c1', 'm1', 'k1']#plt.cm.(np.linspace(0,1,2*len(data_dict))) # since I won't put 10 digits in one plot
        for j, key in enumerate(data_dict):
            data = data_dict[key][i]
            classification = classification_dict[key][i]
            
            normal = data[classification == 1]
            anomalous = data[classification == -1]
            
            # set color
            #color = next(ax._get_lines.prop_cycler)['color']
            
            # plot normal
            x = [i[0] for i in normal]
            y = [i[1] for i in normal]
            plt.scatter(x, y, color=color[j], s=25, linewidths=.8, marker=markers[j], alpha=0.7, zorder=3 ,label=key+' - Normal', edgecolors="k") # , marker=markers[j]

            # plot anomalous
            x = [i[0] for i in anomalous]
            y = [i[1] for i in anomalous]
            plt.scatter(x, y, color=color[j], s=25, linewidths=.8, marker=markers[j], alpha=0.7, zorder=3, label=key+' - Anomalous', edgecolors="r")
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  fancybox=True, shadow=True, ncol=nr_digits) # , prop={'size': 8}
        ax.grid(alpha=0.4, zorder=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.87, top=0.86)
        
        # save figure without and with title
        plt.savefig(f"{file_name}/trial_{i}_all_no_title", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
        plt.title(r"$\overline{h_i}$" + f" Test States {'- Job '+str(job_id) if job_id is not None else ''}- Trial_{i}")
        plt.savefig(f"{file_name}/trial_{i}_all", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
        plt.close('all')


def sort_feature_splitting(data, feature, splittings):
    """
    Takes specific splittings from dataset, options:
    'all', 'first', 'last', 'mean'
    """
    if splittings == 'all':
        return ak.flatten(data[feature])
    elif splittings == 'first':
        return ak.firsts(data[feature])
    elif splittings == 'last':
        return [d[-1] for d in data[feature]]
    elif splittings == 'mean':
        return [ak.mean(d) for d in data[feature]]
    return -1

def normal_vs_anomaly_2D_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, splittings, job_id=None, trial=None):
    # first splitting, last splitting, mean of splittings or all splittings
    #ak.firsts(g_normal[feature]),
    #[[g_normal[feature][i][-1] for i in range(len(g_normal[feature]))],
    #[[ak.mean(x) for x in g_normal[feature]],
    #ak.flatten(g_normal[feature]),
        
    # make out directory if it does not exist yet
    out_dir = f"testing/output/2D_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    data = ((g_anomaly, g_normal), (q_anomaly, q_normal))
    feature_combinations= [
        (features[0], features[1]),
        (features[0], features[2]),
        (features[1], features[2])
    ]
    
    # set colors, same as in other plot
    color = ['b', 'tab:orange', 'g', 'r', 'k', 'b1', 'c1', 'm1', 'k1']#plt.cm.(np.linspace(0,1,2*len(data_dict))) # since I won't put 10 digits in one plot
    markers = ["o", "s", "v", "*" , "D" , "-"]
    labels = ["Gluon Jets", "Quark Jets"]
    
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(1, 3, figsize=(8 * 1.36, 3.3), dpi=160)
    
    #TODO only for 3 features now (because easier)
    for i, (f1, f2) in enumerate(feature_combinations):
        for j,d in enumerate(data): # quark and gluon jets
            
            normal = d[1]
            anomalous = d[0]
            
            # plot normal
            x = sort_feature_splitting(normal, f1, splittings)
            y = sort_feature_splitting(normal, f2, splittings)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            axs[i].scatter(x, y, color=color[2*j], s=3, linestyle='None', marker=markers[j], alpha=0.5, zorder=3 ,label=labels[j]+' - Normal') # , edgecolors="k", marker=markers[j]

            # plot anomalous
            x = sort_feature_splitting(anomalous, f1, splittings)
            y = sort_feature_splitting(anomalous, f2, splittings)
            xmin = np.min(x) if np.min(x) < xmin else xmin
            xmax = np.max(x) if np.max(x) > xmax else xmax
            ymin = np.min(y) if np.min(y) < ymin else ymin
            ymax = np.max(y) if np.max(y) > ymax else ymax
            axs[i].scatter(x, y, color=color[2*j+1], s=3,  linestyle='None', marker=markers[j], alpha=0.7, zorder=4, label=labels[j]+' - Anomalous') #linewidths=.6,
            
            # setup
            axs[i].set_xlim([xmin, xmax])
            axs[i].set_ylim([ymin, ymax])
            axs[i].set_xlabel(na.variable_names[f1])
            axs[i].set_ylabel(na.variable_names[f2])
            axs[i].grid(alpha=0.4, zorder=0)


    # Shrink current axis's height by 10% on the bottom
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    lgd = axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
                fancybox=True, shadow=True, ncol=len(labels)) # , prop={'size': 8}
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.87, top=0.86, wspace=0.3, hspace=0.4)
    
    # save figure without and with title #TODO title
    plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}{splittings}_no_title", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
    plt.title(r"$\overline{h_i}$" + f" Test States {'- Job '+str(job_id) if job_id is not None else ''}- Trial_{i}")
    plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}{splittings}", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
    plt.close('all')
    return


def sk_train_plot(model, X1, y=None, fit=False, ax=plt):
    if fit:
        # fit (train) and predict the data, if y
        model.fit(X1, y, sample_weight=None) # TODO figure out sample_weight
        
    pred = model.predict(X1)
    
    # meshgrid for plots
    xx1, yy1 = np.meshgrid(np.linspace(X1[:,0].min(), X1[:,0].max(), 500),
                        np.linspace(X1[:,1].min(), X1[:,1].max(), 500))
    
    # decision function
    Z1 = model.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    
    # plot data and decision function
    ax.scatter(X1[:, 0], X1[:, 1], c=y, cmap=plt.cm.viridis, alpha=0.25)
    ax.contour( xx1, yy1, Z1, levels=(-1,0,1), linewidths=(1, 1, 1),
                linestyles=('--', '-', '--'), colors=('b','k', 'r'))
    
    # Plot support vectors (non-zero alphas)
    # as circled points (linewidth > 0)
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c=y[model.support_],
                cmap=plt.cm.viridis, lw=1, edgecolors='k')
    return



# roc curve function
def ROC_curve_digits(digits_data_normal, digits_data_anomaly, trials, job_id):
    
    # store roc curve plots in designated directory
    out_dir = f"testing/output/ROC_curves"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # mock arrays for moniker 1 or 0 if gluon or quark
    normal_true = [{"y_true": 1} for _ in range(len(digits_data_normal))]
    anomaly_true =[{"y_true": 0} for _ in range(len(digits_data_anomaly))]

    collect_aucs = []

    for i in range(len(trials)):
        # mix recur dataset with labels
        data_list = [{"data": item, "y_true": 1} for item in digits_data_normal] + [{"data": item, "y_true": 0} for item in digits_data_anomaly]
        
        # select model
        result = trials[i]["result"]
        
        # get models
        lstm_model = result["model"]["lstm"]  # note in some old files it is lstm:
        ocsvm_model = result["model"]["ocsvm"]
        scaler = result["model"]["scaler"]
        
        # get important parameters
        batch_size = result['hyper_parameters']['batch_size']
        pooling = result['hyper_parameters']['pooling']
        final_cost = result['final_cost']
        
        ### build a single branch from all test data ###
        data = [d["data"] for d in data_list]
        data, batch_size, track_jets_data, _ = single_branch(data)
        ###########

        data_loader = lstm_data_prep(
            data=data,
            scaler=scaler,
            batch_size=batch_size,
        )

        input_dim = len(data[0])

        # get h_bar states
        h_bar_list, _, _ = calc_lstm_results(
            lstm_model,
            input_dim,
            data_loader,
            track_jets_data,
            pooling=pooling,
        )
        h_bar_list_np = h_bar_list_to_numpy(h_bar_list)
        
        # get decision function results
        y_predict = ocsvm_model.decision_function(h_bar_list_np)
        data_list = [ {**item, "y_predict":y} for item, y in zip(data_list, y_predict)]
        
        # sort by y_predict
        data_list = sorted(data_list, key=lambda d: d['y_predict'])
        
        y_predict = [x["y_predict"] for x in data_list]
        y_true = [d['y_true'] for d in data_list]
        plot_title = f"ROC Curve Digits - Job: {job_id}"
        if len(trials) > 1:
            print(f"\nTrial {i}:\nWith final cost: {final_cost:.2E}")
            plot_title += f"- Trial {i}"
        out_file = out_dir + "/ROC_curve_trial" + str(i)
        _, auc = ROC_plot_curve(y_true, y_predict, plot_title, out_file)
        collect_aucs.append(auc)
    
    return collect_aucs


def get_dr_kt(branches, features=["sigJetRecur_dr12", "sigJetRecur_jetpt", "sigJetRecur_z"]):
    """Obtain flattened dr and kt array for Lund plane
    default features are recursive, can be changed for normal softdrop:
    order = [dr, pt, z]"""
    try:
        flat_dr = ak.flatten(ak.flatten(branches[features[0]]))
        flat_pt = ak.flatten(ak.flatten(branches[features[1]]))
        flat_z = ak.flatten(ak.flatten(branches[features[2]]))
    except:
        flat_dr = ak.flatten(branches[features[0]])
        flat_pt = ak.flatten(branches[features[1]])
        flat_z = ak.flatten(branches[features[2]])
    #flat_kt = (1-flat_z) * flat_pt * flat_dr
    flat_kt = flat_z * flat_pt * flat_dr
    return flat_dr, flat_kt


def lund_planes(normal, anomalous, job_id=None, trial=None, labels = ["Normal Data", "Anomalous Data"], info="anomalies", R=0.4):
    """Normal data vs anomaly data, next to each other, difference, other?"""
    
    # store roc curve plots in designated directory
    out_dir = f"testing/output/Lund"
    if job_id: out_dir += f"_{job_id}" 
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    # create histograms
    histograms = []
    for dataset in (normal, anomalous):
        flat_dr, flat_kt = get_dr_kt(dataset)
        x = np.log(R/flat_dr)
        y = np.log(flat_kt)
        histograms.append(np.histogram2d(x.to_numpy(), y.to_numpy(), range=[[0, 8], [-7, 5]], bins=20))
    vmin = min([np.min(H[H > 0]) for H, _, _ in histograms])
    vmax = max([np.max(H) for H, _, _ in histograms])
        
    # create plot
    plt.rcParams.update({'font.size': 24})
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize = (12,5),dpi=160)#, gridspec_kw={'width_ratios': [1, 1.5]})
    fig.patch.set_facecolor('white')

    for i, (ax, histogram) in enumerate(zip(axs.flat, histograms)):
        H, xedges, yedges = histogram
        im = ax.imshow(H.T,
                        interpolation='nearest',
                        origin='lower', 
                        norm=colors.LogNorm(vmin=vmin,vmax=vmax), 
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        aspect=0.7
                    )
        ax.title.set_text(labels[i])
        ax.set_xlabel(r'$\ln (R/\Delta R)$')
        ax.set_ylabel(r'$\ln (k_t)$')
        ax.set_xticks(np.arange(xedges[0], xedges[-1] +2, 2.0))
        ax.set_yticks(np.arange(min(yedges+1), max(yedges), 2.0) if 0 not in yedges else np.arange(min(yedges), max(yedges), 2.0))

    # colorbar & adjust
    fig.subplots_adjust(left=0.12, right=.92, top=0.9, bottom=0.18, hspace=0.3, wspace=0.4)
    im_ratio = H.shape[0]/H.shape[1] 
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.046*im_ratio, pad=0.04, shrink=0.99) #  ,shrink=0.65
    cbar.set_label("Count")

    # savefig
    plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}_{info}")
    plt.close('all')

    """
    # - difference -
    fig = plt.figure( figsize = (18, 5))#, gridspec_kw={'width_ratios': [1, 1.5]})
    fig.patch.set_facecolor('white')

    # First dataset zcut = 0.0
    H3 = H2 - H
    xedges3 = xedges2 - xedges
    yedges3 = yedges2 - yedges
    plt.imshow(H3.T, interpolation='nearest', origin='lower', norm=colors.LogNorm(), extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]], aspect=0.7)

    plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}_anomalies_difference")
    """
    return

def lund_planes_anomalies_qg(g_anomaly, g_normal, q_anomaly, q_normal, job_id, trial=None, R=0.4):
    """Normal data vs anomaly data, next to each other, difference, other?"""
    
    # store roc curve plots in designated directory
    out_dir = f"testing/output/Lund"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    normal = ak.concatenate((g_normal,q_normal))
    anomalous = ak.concatenate((g_anomaly,q_anomaly))
    lund_planes(normal, anomalous, job_id, trial, R=R)
    
    labels = ["Normal Gluon Data", "Anomalous Gluon Data", "Normal Quark Data", "Anomalous Quark Data"]

    # create histograms
    histograms = []
    for dataset in (g_normal, g_anomaly, q_normal, q_anomaly):
        flat_dr, flat_kt = get_dr_kt(dataset)
        x = np.log(R/flat_dr)
        y = np.log(flat_kt)
        histograms.append(np.histogram2d(x.to_numpy(), y.to_numpy(), range=[[0, 8], [-7, 5]], bins=20))
    vmin = min([np.min(H[H > 0]) for H, _, _ in histograms])
    vmax = max([np.max(H) for H, _, _ in histograms])
        
    # create plot
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize = (12,10),dpi=160)#, gridspec_kw={'width_ratios': [1, 1.5]})
    fig.patch.set_facecolor('white')

    for i, (ax, histogram) in enumerate(zip(axs.flat, histograms)):
        H, xedges, yedges = histogram
        im = ax.imshow(H.T,
                        interpolation='nearest',
                        origin='lower', 
                        norm=colors.LogNorm(vmin=vmin,vmax=vmax), 
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        aspect=0.7
                    )
        ax.title.set_text(labels[i])
        ax.set_xlabel(r'$\ln (R/\Delta R)$')
        ax.set_ylabel(r'$\ln (k_t)$')
        ax.set_xticks(np.arange(xedges[0], xedges[-1] +2, 2.0))
        ax.set_yticks(np.arange(min(yedges+1), max(yedges), 2.0) if 0 not in yedges else np.arange(min(yedges), max(yedges), 2.0))

    # colorbar & adjust
    fig.subplots_adjust(left=0.05, right=.9, top=0.95, bottom=0.1, hspace=0.4, wspace=0.2)
    im_ratio = H.shape[0]/H.shape[1] 
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), aspect=25, fraction=0.046*im_ratio, pad=0.08, shrink=0.85) #  ,shrink=0.65
    cbar.set_label("Count")

    # savefig    
    plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}_anomalies_qg")
    plt.close('all')
    
    # # - difference -
    # fig = plt.figure( figsize = (18, 5))#, gridspec_kw={'width_ratios': [1, 1.5]})
    # fig.patch.set_facecolor('white')

    # # First dataset zcut = 0.0
    # H3 = H2 - H
    # xedges3 = xedges2 - xedges
    # yedges3 = yedges2 - yedges
    # plt.imshow(H3.T, interpolation='nearest', origin='lower', norm=colors.LogNorm(), extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]], aspect=0.7)

    # plt.savefig(f"{out_dir}/{'trial_'+str(trial)+'_' if type(trial) == int else ''}_anomalies_difference_qg")

    return

