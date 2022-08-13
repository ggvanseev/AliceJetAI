import numpy as np
import matplotlib.pyplot as plt

# per variable
auc_scores_hand = {
    "sigJetRecur_dr12": [0.73],
    "sigJetRecur_z": [0.51],
    "sigJetRecur_jetpt": [0.49],
}

# per job per dimension
auc_scores_lstm = {
    "22_08_09_1909": [0.47, 0.47, 0.53],
    "22_08_09_1934": [0.60, 0.62, 0.59],
    "22_08_09_1941": [0.49, 0.58, 0.40],
    "22_08_11_1520": [0.68, 0.28, 0.44]
}

# per job
auc_scores_lstm_ocsvm = {
    "22_08_09_1909": [0.49],
    "22_08_09_1934": [0.35],
    "22_08_09_1941": [0.55],
    "22_08_11_1520": [0.42]
}

bins = 20
plt.figure()
plt.title("AUC Scores of ROC Curves")
plt.hist([y for x in list(auc_scores_hand.values()) for y in x], bins=bins, stacked=True, label="Hand Cut Variables")
plt.hist([y for x in list(auc_scores_lstm.values()) for y in x], bins=bins, stacked=True, label="Hand Cut LSTM Hidden State")
plt.hist([y for x in list(auc_scores_lstm_ocsvm.values()) for y in x], bins=bins, stacked=True, label="LSTM + OCSVM")
plt.legend()
plt.show()