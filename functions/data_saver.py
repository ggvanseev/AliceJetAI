import matplotlib.pyplot as plt
import torch
import json


def save_loss_plots(step_training, loss_training, i_trial, i, loss_func):
    plt.clf()
    plt.plot(step_training, loss_training)
    plt.xlabel("Step Index")
    plt.ylabel("Loss")
    if loss_func == "mse":
        plt.ylim(0.3, 0.6)
        plt.savefig("./loss/MSE_{}_{}.png".format(i_trial, i))
    if loss_func == "bce":
        plt.ylim(0.8, 1.5)
        plt.savefig("./loss/BCE_{}_{}.png".format(i_trial, i))


def save_results(prefix, model, hyper_parameters):
    model_path = "./model/" + prefix + ".pt"
    torch.save(model.state_dict(), model_path)
    json_path = "./model/" + prefix + ".json"
    with open(json_path, "w") as json_file:
        json.dump(hyper_parameters, json_file, indent=4)


class DataTrackerTrials:
    i_trial: int = 0
    index_trial: list = []
    loss_trial: list = []
