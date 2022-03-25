import matplotlib.pyplot as plt

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
    
    return 


def normal_vs_anomaly_2D_all(data, classification, file_name):
    first_key = list(data.keys())[0]
    nr_trials = len(data[first_key])
    
    for i in range(nr_trials):
        pass
    
    
    return
    