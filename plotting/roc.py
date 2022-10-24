import awkward as ak
import matplotlib as mpl
import matplotlib.pylab as plt
mpl.rcParams.update(mpl.rcParamsDefault)
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import os

from torch import threshold

from functions.run_lstm import calc_lstm_results
from functions.data_manipulation import (
    lstm_data_prep,
    branch_filler,
    h_bar_list_to_numpy,
    format_ak_to_list,
    single_branch,
)
import branch_names as na

def ROC_zero_division(pos, neg):
    """Check if positive is 0, otherwise return ratio"""
    if (pos == 0):
        return 0
    return pos / (pos + neg)

def ROC_plot_ax(y_true:list, y_predict:list):
    
    # set font size
    plt.rcParams.update({'font.size': 16})
    
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    print(f"ROC Area under curve: {roc_auc:.4f}")
    
    ax.plot(fpr, tpr, color="C1", label="Sklearn Metrics") 
    ax.plot([0,1],[0,1],color='k')
    return ax


def ROC_plot_curve(y_true:list, y_predict:list, plot_title:str, out_file:str, xlabel="False Positive Rate", ylabel="True Positive Rate") -> plt.figure:
    """Function that plots a ROC curve from true and predicted 
    labels for data

    Args:
        y_true (list): true labels
        y_predict (list): predicted labels
        plot_title (str): title for the ROC curve plot
        out_file (str): output file name

    Returns:
        plt.figure: ROC curve plot
    """    
    
    # set font size
    plt.rcParams.update({'font.size': 16})
    
    # get minimum and maximum values
    minimum = min(y_predict)
    maximum = max(y_predict)
    
    tpr = list() # True Positive Rate
    fpr = list() # False Positive Rate
    results = np.array([y_true, y_predict]).T
    for j in np.linspace(minimum, maximum, 10000):
        false_neg_under_th = np.count_nonzero(results[(results[:,0] == 1) & (results[:,1] < j)])
        true_pos_under_th = np.count_nonzero(results[(results[:,0] == 1) & (results[:,1] >= j)])
        
        true_neg_under_th = np.count_nonzero(results[(results[:,0] == 0) & (results[:,1] < j)])
        false_pos_under_th = np.count_nonzero(results[(results[:,0] == 0) & (results[:,1] >= j)])
    
        tpr.append(ROC_zero_division(true_pos_under_th, false_neg_under_th))
        fpr.append(ROC_zero_division(false_pos_under_th, true_neg_under_th))
            
    # make plot
    fig, ax = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
    #ax.plot(fpr, tpr, label="Own Code") TODO
    
    # TODO using sklearn metrics -> Tested to be same as above, but requires less points for ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    print(f"ROC Area under curve: {roc_auc:.4f}")
    try:
         print(f"Model accuracy: {accuracy_score(y_true, np.sign(y_predict))}")
    except:
        pass

    roc_plot = ax.plot(fpr, tpr, color="C1", linewidth=2, zorder=3, label="Sklearn Metrics")
    roc_plot[0].set_clip_on(False) 
    ax.plot([0,1],[0,1], alpha=0.7, color='k', label="Random Classifier")
    
    # plot textbox
    if roc_auc <= 0.5:
        x = 0.35
        y = 0.7
    else:
        x = 0.7
        y = 0.35
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.8)
    ax.text(x, y, f"Area Under Curve: {roc_auc:.4f}", ha="center", va="center", size=17, bbox=bbox_props)
    
    # set plot values
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(alpha=0.4)
    #ax.legend() TODO
    
    # Hide the right and top spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # save plot without title
    plt.savefig(out_file+"_no_title.png")
    
    # save plot with title
    plt.title(plot_title, y=1.04)
    plt.savefig(out_file+".png")
    print(f"ROC curve stored at:\n\t{out_file}")
    
    return fig, roc_auc
    #plt.close()  # close figure - clean memory


# roc curve function
def ROC_curve_qg(g_recur, q_recur, trials, job_id):
    
    # store roc curve plots in designated directory
    out_dir = f"output/ROC_curves"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    # mock arrays for moniker 1 or 0 if gluon or quark
    g_true = ak.Array([{"y_true": 1} for _ in range(len(g_recur))])
    q_true = ak.Array([{"y_true": 0} for _ in range(len(q_recur))])
    
    collect_aucs = []
    
    for i in range(len(trials)):
        # mix recur dataset with labels
        data_list = [{**item, **y} for item, y in zip(g_recur.to_list(), g_true.to_list())] + [{**item, **y} for item, y in zip(q_recur.to_list(), q_true.to_list())]
    
        # select model
        result = trials[i]["result"]
        
        # get models
        lstm_model = result["model"]["lstm"]  # note in some old files it is lstm:
        ocsvm_model = result["model"]["ocsvm"]
        scaler = result["model"]["scaler"]
        
        # get important parameters
        batch_size = result['hyper_parameters']['batch_size']
        pooling = result['hyper_parameters']['pooling']
        input_variables = list(result['hyper_parameters']["variables"])
        final_cost = result['final_cost']
    
        # reformat data to go into lstm
        data = format_ak_to_list([{ key: d[key] for key in input_variables } for d in data_list])
        data = [x for x in data if len(x[0]) > 0] # remove empty stuff
        
        ### build a single branch from all test data ###
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
        plot_title = f"ROC Curve Gluon vs Quark Jets - Job: {job_id}"
        if len(trials) > 1:
            print(f"\nTrial {i}:\nWith final cost: {final_cost:.2E}")
            plot_title += f"- Trial {i}"
        out_file = out_dir + "/ROC_curve_trial" + str(i)
        _, auc = ROC_plot_curve(y_true, y_predict, plot_title, out_file, xlabel="Normal Fraction Quarks (FPR)", ylabel="Normal Fraction Gluons (TPR)")
        collect_aucs.append(auc)
        plt.close('all')
    
    return collect_aucs

    
def ROC_feature_curve_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, samples=None):
    """Make ROC curve for given features for quarks and gluons
    
    TODO I cannot wrap my head around why I made this, it does
    not make any sense to me anymore, leave it for now, but it's not good
    TODO probably delete it
    """
    
    # store stacked plots in designated directory
    out_dir = f"output/ROC_curves"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    for feature in features:
        # check what kind of sample to take e.g. first-, last-, all splittings
        if samples == "first":
            true_pos = ak.firsts(g_normal[feature])
            false_neg = ak.firsts(g_anomaly[feature])
            false_pos = ak.firsts(q_normal[feature])
            true_neg = ak.firsts(q_anomaly[feature])
        elif samples == "last":
            true_pos = ak.Array([g_normal[feature][i][-1] for i in range(len(g_normal[feature]))])
            false_neg = ak.Array([g_anomaly[feature][i][-1] for i in range(len(g_anomaly[feature]))])
            false_pos = ak.Array([q_normal[feature][i][-1] for i in range(len(q_normal[feature]))])
            true_neg = ak.Array([q_anomaly[feature][i][-1] for i in range(len(q_anomaly[feature]))])
        else:
            true_pos = g_normal[feature]
            false_neg = g_anomaly[feature]
            false_pos = q_normal[feature]
            true_neg = q_anomaly[feature]
        
        # get minimum and maximum values
        minimum = ak.min((false_neg, true_pos, true_neg, false_pos))
        maximum = ak.max((false_neg, true_pos, true_neg, false_pos))
        
        tpr = list() # True Positive Rate
        fpr = list() # False Positive Rate
        for i in np.linspace(minimum, maximum, 100):
            false_neg_under_th = ak.count_nonzero(false_neg[false_neg < i]) # false negative
            true_pos_under_th = ak.count_nonzero(true_pos[true_pos < i]) # true positive
            
            true_neg_under_th = ak.count_nonzero(true_neg[true_neg < i]) # true negative
            false_pos_under_th = ak.count_nonzero(false_pos[false_pos < i]) # false positive
            
            tpr.append(ROC_zero_division(true_pos_under_th, false_neg_under_th))
            fpr.append(ROC_zero_division(false_pos_under_th, true_neg_under_th))
        splitting_string = samples + " splittings" if samples else "all splittings"
        plt.title(f"ROC curve Gluon vs Quark {splitting_string} - {feature} - {job_id}")
        plt.scatter(fpr, tpr)
        plt.plot([0,1],[0,1],color='k')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlabel("Normal Fraction Gluons")
        plt.ylabel("Normal Fraction Quarks")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # save plot
        splitting_string = samples + "_splittings" if samples else "all_splittings"
        plt.savefig(out_dir + "/ROC_curve_" + splitting_string + feature)
        plt.close('all')  # close figure - clean memory
            
    return


def ROC_anomalies_hand_cut(g_recur, q_recur, cut_variable, pooling="first"):
    
    # print text
    print(f"\nFor variable {na.variable_names[cut_variable]}:")
    
    # store stacked plots in designated directory
    out_dir = f"output/ROC_curves_hand_cuts"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    # mock arrays for moniker 1 or 0 if gluon or quark
    g_true = ak.Array([{"y_true": 1} for i in range(len(g_recur))])
    q_true = ak.Array([{"y_true": 0} for i in range(len(q_recur))])
    data_list = [{**item, **y} for item, y in zip(g_recur.to_list(), g_true.to_list())] + [{**item, **y} for item, y in zip(q_recur.to_list(), q_true.to_list())]

    # 1st splitting of the recursive jet data is the SoftDrop splitting -> make cuts on this splitting
    if pooling == "first":
        y_predict = [d[cut_variable][0] for d  in data_list]
    elif pooling == "last":
        y_predict = [d[cut_variable][-1] for d  in data_list]
    elif pooling == "mean":
        y_predict = [ak.mean(d[cut_variable]) for d  in data_list]
    else:
        print("Wrong pooling method given!")
        return
    # Not necessary to extrapolate? -> maybe for the lstm results TODO
    # y_predict = [y - min(y_predict) for y in y_predict]     # move set s.t. lowest value is at 0
    # y_predict = [y / max(y_predict) for y in y_predict] # stretch set s.t. highest value is at 1
    data_list = [ {**item, "y_predict":y} for item, y in zip(data_list, y_predict)]
    y_true = [d['y_true'] for d in data_list]
    
    plot_title = f"ROC Curve Quark & Gluon Jets - Cuts On Variable {na.variable_names[cut_variable]}"
    out_file = f"{out_dir}/on_variable_{cut_variable}_"+pooling

    ROC_plot_curve(y_true, y_predict, plot_title, out_file, xlabel="Normal Fraction Quarks (FPR)", ylabel="Normal Fraction Gluons (TPR)")

    return 


def ROC_anomalies_hand_cut_lstm(g_recur, q_recur, job_id, trials):
    
    # variable list - store for input
    variables = g_recur.fields
    
    # store roc curve plots in designated directory
    out_dir = f"output/ROC_curves"
    out_dir += f"_{job_id}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    # mock arrays for moniker 1 or 0 if gluon or quark
    g_true = ak.Array([{"y_true": 1} for i in range(len(g_recur))])
    q_true = ak.Array([{"y_true": 0} for i in range(len(q_recur))])
    
    # collect auc per trial
    collect_aucs = []
    
    for i in range(len(trials)):
        # mix recur dataset with labels
        data_list = [{**item, **y} for item, y in zip(g_recur.to_list(), g_true.to_list())] + [{**item, **y} for item, y in zip(q_recur.to_list(), q_true.to_list())]
    
        # select model
        result = trials[i]["result"]
        
        # get models
        lstm_model = result["model"]["lstm"]  # note in some old files it is lstm:
        ocsvm_model = result["model"]["ocsvm"]
        scaler = result["model"]["scaler"]
        
        # get important parameters
        batch_size = result['hyper_parameters']['batch_size']
        pooling = result['hyper_parameters']['pooling']
        input_variables = list(result['hyper_parameters']["variables"])
        final_cost = result['final_cost']
    
        # reformat data to go into lstm
        data = format_ak_to_list([{ key: d[key] for key in input_variables } for d in data_list])
        data = [x for x in data if len(x[0]) > 0] # remove empty stuff
    
        # reformat data to go into lstm
        data = format_ak_to_list([{ key: d[key] for key in input_variables } for d in data_list])
        data = [x for x in data if len(x[0]) > 0] # remove empty stuff
        
        ### build a single branch from all test data ###
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
        
        dimensions = h_bar_list_np.shape[1]
        
        # store aucs of trial
        aucs = []
        
        # make ROC curve for each dimension
        for j in range(dimensions):
            
            # print text
            print(f"For trial {i} and dimension {j}:")
        
            # 1st splitting of the recursive jet data is the SoftDrop splitting -> make cuts on this splitting
            y_predict = h_bar_list_np[:, j]
        
            # TODO cut variable is mean pooled/last pooled -> already happened in calc_lstm_results
            data_list = [ {**item, "y_predict":y} for item, y in zip(data_list, y_predict)]
            y_true = [d['y_true'] for d in data_list]

            plot_title = f"ROC Curve on LSTM results - Manually Cut Dimension {j}"
            if len(trials) > 1:
                plot_title += f" - Trial {i}"
            out_file = f"{out_dir}/trial_{i}_on_dimension_{j}"
            _, auc = ROC_plot_curve(y_true, y_predict, plot_title, out_file, xlabel="Normal Fraction Quarks", ylabel="Normal Fraction Gluons")
            aucs.append(auc)
        collect_aucs.append(aucs)
    return collect_aucs