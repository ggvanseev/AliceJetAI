import matplotlib.pyplot as plt
import time

def plot_cost_vs_cost_condition(track_cost, track_cost_condition,title_plot, show_flag=False,save_flag=False):
    
    fig, ax1 = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
    fig.suptitle(title_plot, y=1.08)
    ax1.plot(track_cost_condition[1:])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cost Condition")

    ax2 = ax1.twinx()
    ax2.plot(track_cost[1:], "--", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Cost")
    plt.legend()

    if save_flag:
        fig.savefig("output/" + title_plot + str(time.time()) + ".png")
    if show_flag:
        plt.show()
