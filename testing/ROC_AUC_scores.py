from encodings import normalize_encoding
from locale import normalize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# per variable
auc_scores_hand = {
    "sigJetRecur_dr12": [0.73],
    "sigJetRecur_z": [0.51],
    "sigJetRecur_jetpt": [0.49],
}

# per job per dimension TODO result for every trial for every hidden dim
auc_scores_lstm = {
    "22_08_09_1909": [[0.47, 0.47, 0.53]],
    "22_08_09_1934": [[0.60, 0.62, 0.59]],
    "22_08_09_1941": [[0.49, 0.58, 0.40]],
    "22_08_11_1520": [[0.68, 0.28, 0.44]]
}

# per job
auc_scores_lstm_ocsvm = {
    "22_07_18_1520": [0.62],
    "22_08_09_1909": [0.49],
    "22_08_09_1934": [0.35],
    "22_08_09_1941": [0.55],
    "22_08_11_1520": [0.42],
    '22_07_18_1327': [0.29389671361502345], 
    '22_07_18_1334': [0.29325920434890046], 
    '22_07_18_1340': [0.3298888065233507], 
    '22_07_18_1345': [0.28823820113664445], 
    '22_07_18_1357': [0.31957499382258464], 
    '22_07_18_1404': [0.42499135161848284], 
    '22_07_18_1410': [0.3892760069187052], 
    '22_07_18_1414': [0.3968173956016802], 
    '22_07_18_1417': [0.39421299728193726], 
    '22_07_18_1432': [0.4506597479614529], 
    '22_07_18_1435': [0.4375982209043736], 
    '22_07_18_1440': [0.4157499382258463], 
    '22_07_18_1445': [0.429424264887571], 
    '22_07_18_1452': [0.4341586360266865], 
    '22_07_18_1502': [0.4204348900420064], 
    '22_07_18_1508': [0.5465678280207561], 
    '22_07_18_1514': [0.4053471707437608], 
    '22_07_18_1520': [0.6199456387447492], 
    '22_08_09_1909': [0.4948356807511737], 
    '22_08_09_1934': [0.345910551025451], 
    '22_08_09_1941': [0.551830985915493], 
    '22_08_11_1520': [0.4168668149246355],
    '10993304': [0.5371237954040029], 
    '10993305': [0.4154583642204102], 
}

auc_scores_lstm_ocsvm_hypertraining = {
    '11120653': [0.45917963923894245,0.4075463306152706, 0.4293353101062516, 0.7475908080059303, 0.5191648134420559, 0.4802372127501854, 0.5278823820113664, 0.4630985915492958, 0.5128984432913269, 0.5687818136891525, 0.422070669631826, 0.4767185569557697, 0.5611020509019027, 0.568722510501606, 0.5421793921423276, 0.47654558932542623, 0.6796837163330862, 0.5386360266864344, 0.43516679021497406, 0.4681838398813936, 0.3796244131455399, 0.48982456140350883, 0.49778107239930813, 0.46797627872498154, 0.5056634544106746, 0.4827971336792685, 0.47819125277983693, 0.49321966889053614, 0.4925574499629355, 0.5137929330368174, 0.5310699283419817, 0.40343958487768716], 
    '11120654': [0.37694094390906846, 0.5210476896466518, 0.5840820360761058, 0.7074870274277242, 0.6996046454163578, 0.4804200642451198, 0.35522115147022487, 0.5476995305164319, 0.4595552260934025, 0.4822881146528292, 0.5241858166543119, 0.5198715097603163, 0.4416061279960465, 0.681077341240425, 0.4248035581912528, 0.3657721769211762, 0.5657029898690389, 0.4419273535952558, 0.48297010130961204, 0.4731554237706943, 0.43667407956511, 0.600439831974302, 0.5180380528786755, 0.46382011366444276, 0.49292809488510003, 0.4867062021250309, 0.4358537188040524, 0.47353595255744996, 0.6953990610328639, 0.4507437608104769, 0.4507832962688411, 0.6531208302446256, 0.6575537435137138, 0.6940696812453669, 0.5057227575982209, 0.3790511489992587, 0.4826735853718803, 0.45021003212255994, 0.45845317519149986, 0.4968964665184087, 0.37382752656288604, 0.4528440820360761, 0.4948455646157648, 0.36478379046207066, 0.5389424264887571], 
    '11120655': [0.4611069928341982, 0.4702100321225599, 0.33580429948109713, 0.5029157400543612, 0.44941932295527554, 0.5196837163330863, 0.37785025945144546, 0.512656288608846, 0.3178008401284902, 0.6990313812700766, 0.33499382258463056, 0.3587842846553002, 0.4800444773906597, 0.5011860637509266, 0.5011020509019026, 0.5540202619224117, 0.5435829009142575, 0.5127748949839387, 0.6644625648628614, 0.380721522115147, 0.5007165801828515, 0.5232863849765258, 0.39210279219174693, 0.34409686187299227, 0.5944403261675315, 0.4889696071163825, 0.5434890042006424, 0.36294539164813444, 0.46502594514455153, 0.3148307388188782, 0.4665085248332098, 0.3363627378304917, 0.5427674820854954, 0.503953545836422, 0.38967136150234744, 0.3237311588831233, 0.4510798122065728, 0.5800000000000001]
}

kwargs = dict(histtype='stepfilled', density=True, alpha=0.5,  bins=40) # normed=True,

# plt.figure()
# plt.title("AUC Scores of ROC Curves")
# plt.hist([y for x in list(auc_scores_lstm_ocsvm.values()) for y in x], label="LSTM + OCSVM", **kwargs)
# plt.hist([y for x in list(auc_scores_lstm.values()) for y in x], label="Hand Cut LSTM Hidden State", **kwargs)
# plt.hist([y for x in list(auc_scores_hand.values()) for y in x], label="Hand Cut Variables", **kwargs)
# plt.legend()
# plt.show()

dicts = [auc_scores_lstm_ocsvm_hypertraining, auc_scores_lstm_ocsvm, auc_scores_lstm, auc_scores_hand]
labels = ["LSTM + OCSVM - HyperTraining", "LSTM + OCSVM", "Hand Cut LSTM Hidden State", "Hand Cut Variables"]
best_dict = {}

# set font size
plt.rcParams.update({'font.size': 13.5})

# make plots and print best scores
print("Best AUC Scores:")
fig, ax = plt.subplots(figsize=[6 * 1.36, 6], dpi=160)
kwargs = dict(ax=ax, kde=False, element="step",bins=60)
for i, (d, l) in enumerate(zip(dicts,labels)):
    sns.histplot(np.concatenate(list(d.values())).ravel(), color=sns.color_palette()[i] ,label=l, **kwargs)
    max_auc = 0
    job = 0
    trial = 0
    dim = -1
    for j, x in d.items():
        try:    
            max_auc, trial, job = (max(x), x.index(max(x)), j) if max(x) > max_auc else (max_auc, trial, job)
        except TypeError: # in case of hidden dimensions
            max_auc = max([y for z in x for y in z])
            for k, hidden in enumerate(x):
                if max_auc in hidden:
                    dim = hidden.index(max_auc)
                    trial = k
                    job = j
    print(f"For {l}:\nJob {job}{' - Trial '+str(trial) if len(x)>1 else ''}{' - Dimension '+str(dim) if dim>=0 else ''}\nAUC {max_auc}\n") 
    best_dict[job] = trial if dim < 0 else (trial, dim)
plt.xlabel("Area Under Curve")
plt.legend()

# store roc curve plots in designated directory
out_dir = f"output/ROC_curves_best"
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

out_file = out_dir + "/auc_scores" + time.strftime("%y_%m_%d_%H%M")

# save plot without title
plt.savefig(out_file+"_no_title")

# save plot with title
plt.title("AUC Scores For Various Tests", y=1.04)
plt.savefig(out_file)

print("Best dict:\n",best_dict)

"""
plt.figure()
kwargs = dict(multiple="stack", kde=False, element="step",bins=60)
sns.histplot(pd.DataFrame(dict(zip(labels, dicts))).melt(), hue="variable", x="value")#,  **kwargs)
plt.show()
"""
