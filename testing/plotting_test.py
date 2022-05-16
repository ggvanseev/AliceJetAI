from attr import s
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np

def sample_plot(data, idx=None, label=""):
    # random sample if no index is given
    if idx == None:
        idx = int(random.random() * len(data))
        
    sample = data[idx]
    x = [i[0] for i in sample]
    y = [i[1] for i in sample]
    plt.scatter(x, y, label=label)
    plt.xlim(min(sample),max(sample))
    plt.ylim(min(sample),max(sample))


def create_sample_plot(data, idx=None):
    plt.figure()
    sample_plot(data, idx)
    plt.show()
    return
    

def normal_vs_anomaly_2D(data, classification, file_name):
     
    normal = data[classification == 1]
    anomalous = data[classification == -1]

    plt.figure()

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


def normal_vs_anomaly_2D_all(data_dict, classification_dict, ocsvm_list, file_name, y=None):
    first_key = list(data_dict.keys())[0]
    nr_trials = len(data_dict[first_key])
    nr_digits = len(data_dict.keys())
    
    markers = ["o", "s", "v", "*" , "D" , "-"]
    
    for i in range(nr_trials):
        
        fig = plt.figure()
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
        Z1 = model.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        
        # plot data and decision function
        ax.contour(xx1, yy1, Z1, levels=(-1,0,1), linewidths=(0.5, 0.75, 0.5),
                    linestyles=('--', '-', '--'), colors=['k','k','k'])
        ax.contourf(xx1, yy1, Z1, cmap=cm.get_cmap("coolwarm_r"), alpha=0.3, linestyles="None")
        
        
        # plot support vectors
        # plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c=y[model.support_] if y else model.predict(model.support_vectors_),
        #        cmap=plt.cm.viridis, lw=1, edgecolors='k', label="Support Vectors")
        
        color = plt.cm.viridis(np.linspace(0,1,2*len(data_dict))) # since I won't put 10 digits in one plot
        for j, key in enumerate(data_dict):
            data = data_dict[key][i]
            classification = classification_dict[key][i]
            
            normal = data[classification == 1]
            anomalous = data[classification == -1]
            
            # plot normal
            x = [i[0] for i in normal]
            y = [i[1] for i in normal]
            plt.scatter(x, y, color=color[2*j], s=15, linewidths=0.7, alpha=0.7, label=key+'_normal', edgecolors="k") # , marker=markers[j]

            # plot anomalous
            x = [i[0] for i in anomalous]
            y = [i[1] for i in anomalous]
            plt.scatter(x, y, color=color[2*j], s=15, linewidths=0.7, alpha=0.7, label=key+'_anomalous', edgecolors="r")
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        plt.title(r"$\overline{h_i}$" + f" test states - trial_{i}")
        plt.xlabel(r"$\overline{h_{i,1}}$")
        plt.ylabel(r"$\overline{h_{i,2}}$")
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  fancybox=True, shadow=True, ncol=nr_digits, prop={'size': 8})
        plt.savefig(f"{file_name}/trial_{i}_all", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)


def sk_train_plot(model, X1,y=None, fit=False, ax=plt):
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