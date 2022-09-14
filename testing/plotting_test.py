from attr import s
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np
import os

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


def normal_vs_anomaly_2D_all(data_dict, classification_dict, ocsvm_list, file_name, job_id=None, y=None, xlabel=r"$\overline{h_{i,1}}$", ylabel=r"$\overline{h_{i,2}}$"):
    
    # set matplotlib font settings
    plt.rcParams.update({'font.size': 12})
    
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


def normal_vs_anomaly_2D_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id=None):
    
    data = ((g_anomaly, g_normal), (q_anomaly, q_normal))
    
    # set colors, same as in other plot
    color = ['b', 'tab:orange', 'c', 'm', 'k', 'b1', 'c1', 'm1', 'k1']#plt.cm.(np.linspace(0,1,2*len(data_dict))) # since I won't put 10 digits in one plot
    markers = ["o", "s", "v", "*" , "D" , "-"]
    
    #TODO only for 3 features now (because easier)
    for i in range(3):
        
        fig = plt.figure(figsize=[6 * 1.36, 6], dpi=160)
        ax = plt.subplot(111)
        
        for j,d in enumerate(data): # quark and gluon jets
            
            normal = d[1]
            anomalous = d[0]
            
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


def lund_planes(g_anomaly, g_normal, q_anomaly, q_normal, job_id):
    
    return