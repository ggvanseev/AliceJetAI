import matplotlib.pyplot as plt
import torch
import json


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
