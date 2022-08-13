import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

def ROC_zero_division(pos, neg):
    """Check if positive is 0, otherwise return ratio"""
    if (pos == 0):
        return 0
    return pos / (pos + neg)

# roc curve function
def ROC_curve_qg(g_recur, q_recur, trials, job_id):
    
    # combine g en q recurs to 1 set -> jets_recur = full set of jets with y_true as q/g moniker
    # get decision values y_predict from decision function for the complete set
    # sort set by decision values
    # split into histogram with 100 bars
    # make ROC plot of decision value vs q/g moniker
    
    # check if needed later! TODO
    #g_recur = format_ak_to_list(g_recur)
    #q_recur = format_ak_to_list(g_recur)
    
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
    
    for i in range(len(trials)):
        # mix 90% g vs 10% q of 1500
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
    
        # reformat data to go into lstm
        data = format_ak_to_list([{ key: d[key] for key in variables } for d in data_list])
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
        results = np.array([[x["y_true"] for x in data_list],[x["y_predict"] for x in data_list]]).T
        false_neg = results[(results[:,0] == 1) & (results[:,1] < 0)]  # g anomaly
        true_pos =  results[(results[:,0] == 1) & (results[:,1] >= 0)] # g normal
        true_neg =  results[(results[:,0] == 0) & (results[:,1] < 0)]  # q anomaly
        false_pos = results[(results[:,0] == 0) & (results[:,1] >= 0)] # q normal
        
        ### plot ROC curve ###
        # get minimum and maximum values
        minimum = min(y_predict)
        maximum = max(y_predict)
        
        tpr = list() # True Positive Rate
        fpr = list() # False Positive Rate
        for j in np.linspace(minimum, maximum, 100):
            false_neg_under_th = ak.count_nonzero(false_neg[false_neg < j]) # false negative
            true_pos_under_th = ak.count_nonzero(true_pos[true_pos < j]) # true positive
            
            true_neg_under_th = ak.count_nonzero(true_neg[true_neg < j]) # true negative
            false_pos_under_th = ak.count_nonzero(false_pos[false_pos < j]) # false positive
            
            tpr.append(ROC_zero_division(true_pos_under_th, false_neg_under_th))
            fpr.append(ROC_zero_division(false_pos_under_th, true_neg_under_th))
        
        # TODO using sklearn metrics    
        fpr, tpr, _ = roc_curve([d['y_true'] for d in data_list], [d['y_predict'] for d in data_list])
        roc_auc = auc(fpr, tpr)
        print(f"ROC: Area under curve: {roc_auc:.2f}")
        
        plt.title(f"ROC curve Gluon vs Quark trial {i} - {job_id}")
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],color='k')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlabel("Normal Fraction Gluons")
        plt.ylabel("Normal Fraction Quarks")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # save plot
        plt.savefig(out_dir + "/ROC_curve_trial" + str(i))
        plt.close()  # close figure - clean memory
    
    
    return
    
def ROC_feature_curve_qg(g_anomaly, g_normal, q_anomaly, q_normal, features, job_id, samples=None):
    """Make ROC curve for given features for quarks and gluons"""
    
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
        plt.close()  # close figure - clean memory
            
    return


def ROC_anomalies_hand_cut(g_recur, q_recur, cut_variable):
    
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
    y_predict = [d[cut_variable][0] for d  in data_list]
    # Not necessary to extrapolate? -> maybe for the lstm results TODO
    #y_predict = [y - min(y_predict) for y in y_predict]     # move set s.t. lowest value is at 0
    #y_predict = [y / max(y_predict) for y in y_predict] # stretch set s.t. highest value is at 1
    data_list = [ {**item, "y_predict":y} for item, y in zip(data_list, y_predict)]
    
    fpr, tpr, _ = roc_curve([d['y_true'] for d in data_list], [d['y_predict'] for d in data_list])
    roc_auc = auc(fpr, tpr)
        
    plt.title(f"ROC Curve - Manually Cut Variable {cut_variable}")
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],color='k')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlabel("Normal Fraction Quark Jets")
    plt.ylabel("Normal Fraction Gluon Jets")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # save plot
    out_file = f"{out_dir}/on_variable_{cut_variable}"
    plt.savefig(out_file)
    plt.close()  # close figure - clean memory
    
    # print text
    print(f"For variable {cut_variable}, ROC curve stored at:\n\t{out_file}")
    print(f"ROC: Area under curve: {roc_auc:.2f}")

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
    
    for i in range(len(trials)):
        # mix 90% g vs 10% q of 1500
        data_list = [{**item, **y} for item, y in zip(g_recur.to_list(), g_true.to_list())] + [{**item, **y} for item, y in zip(q_recur.to_list(), q_true.to_list())]
    
        # select model
        result = trials[i]["result"]
        
        # get models
        lstm_model = result["model"]["lstm"]  # note in some old files it is lstm:
        scaler = result["model"]["scaler"]
        
        # get important parameters
        batch_size = result['hyper_parameters']['batch_size']
        pooling = result['hyper_parameters']['pooling']
    
        # reformat data to go into lstm
        data = format_ak_to_list([{ key: d[key] for key in variables } for d in data_list])
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
        
        # make ROC curve for each dimension
        for j in range(dimensions):
        
            # 1st splitting of the recursive jet data is the SoftDrop splitting -> make cuts on this splitting
            y_predict = h_bar_list_np[:, j]
        
            # TODO cut variable is mean pooled/last pooled -> already happened in calc_lstm_results
            data_list = [ {**item, "y_predict":y} for item, y in zip(data_list, y_predict)]
    
            fpr, tpr, _ = roc_curve([d['y_true'] for d in data_list], [d['y_predict'] for d in data_list])
            roc_auc = auc(fpr, tpr)
                
            plt.title(f"ROC Curve on LSTM results - Manually Cut Dimension {j}")
            plt.plot(fpr, tpr)
            plt.plot([0,1],[0,1],color='k')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.xlabel("Normal Fraction Quark Jets")
            plt.ylabel("Normal Fraction Gluon Jets")
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # save plot
            out_file = f"{out_dir}/on_dimension_{j}"
            plt.savefig(out_file)
            plt.close()  # close figure - clean memory

            # print text
            print(f"For dimension {j}, ROC curve stored at:\n\t{out_file}")
            print(f"ROC: Area under curve: {roc_auc:.2f}")

     
    return 