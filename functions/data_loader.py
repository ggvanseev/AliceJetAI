"""
Loading in data from a ROOT file using uproot. 

load_n_filter data will give the gluon and quark data from SoftDropped and Recursive 
SoftDropped dataset of specified ROOT file. Data is later filtered according to jet 
initiator PDG, eta and pt cuts. Output is in four Awkward arrays: two for quark and
gluon jet data and two for their recursive counterparts.


"""
import uproot
import awkward as ak
import numpy as np
import random
import torch
from typing import Tuple

import branch_names as na
from functions.classification import CLASSIFICATION_CHECK


def select_non_empty_branches(branches, non_empty_key, branches_non_recur=False):
    """
    non_empty_key: give along a recur, because due to zcut a branch might have become empty
    """
    # First filter all comp letely empty branches
    branch_reference = branches[non_empty_key]
    non_empty = list()
    for i in range(len(branch_reference)):
        if len(branch_reference[i]) > 0 and ak.any(branch_reference[i]):
            non_empty.append(i)

    branches = branches[non_empty]
    branches_non_recur = branches_non_recur[non_empty] if branches_non_recur is not False else False
    branch_reference = branches[non_empty_key]

    # empty partially empty entries
    mask = ak.ArrayBuilder()
    for i in range(len(branch_reference)):
        mask.begin_list()
        for j in range(len(branch_reference[i])):
            if len(branch_reference[i, j]) > 0 and ak.any(branch_reference[i, j]):
                mask.integer(j)
        mask.end_list()

    for field in branches.fields:
        branches[field] = branches[field][mask]
        
    if branches_non_recur is not False:
        for field in branches_non_recur.fields:
            branches_non_recur[field] = branches_non_recur[field][mask]
    else:
        branches_non_recur = None   
        
    return branches, branches_non_recur


def concatenate_function(value):
    concatenated_value = ak.concatenate(value, axis=0)
    return concatenated_value


def flatten_array(branches, step_size=1000):
    """
    returns a flattend array
    """
    new_branches = dict()

    for field in branches.fields:
        new_branches[field] = ak.flatten(branches[field])

    return ak.Array(new_branches)


def flatten_array_old(branches):
    """
    returns a flattend array
    """
    new_branches = dict()

    for field in branches.fields:
        new_branches[field] = ak.concatenate(branches[field], axis=0)

    return ak.Array(new_branches)


def load_n_filter_data(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    kt_cut: float = None,
    dr_cut: float = None,
    eta_max: float = 2.0,
    pt_min: int = 130,
    jet_branches: list = None,
    jet_recur_branches: list = [na.recur_dr, na.recur_jetpt, na.recur_z],
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """Load in dataset from ROOT file of jet data. Subsequently, the jet data will be
    split into sets for quarks and gluons as well as sets with recursive jet data for
    quarks and gluons. Therefore, the provided dataset is required to contain the
    variables: jetpt, jet_eta, jet_phi, jet_M, jet_area, recur_splits & parton_match_id.
    Exact labels are given in 'names.py'.

    Args:
        file_name (str): File name of ROOT dataset.
        tree_name (str, optional): Name of the TTree object of the ROOT file. Defaults
                to "jetTreeSig".
        cut (bool, optional): Boolean statement whether to apply cuts. Defaults to True.
        kt_cut (float, optional): Functions as boolean statement whether to apply kt_cut
                to splittings by setting a value. Value for kt cut measured in GeV, sets
                a minumum. Defaults to None, i.e. no kt_cut.
        eta_max (float, optional): Value for eta cut. Defaults to 2.0.
        pt_cut (int, optional): Value for pt cut. Defaults to 130.

    Returns:
        Tuple[ak.Array, ak.Array, ak.Array, ak.Array]: Tuple containing all gluon and
                quark datasets.
    """

    # open file and select jet data and recursive jet data
    branches = uproot.open(file_name)[tree_name].arrays()
    jets_eta_pt_cut = branches[[na.jetpt, na.jet_eta,]]
    jets_recur = branches[jet_recur_branches]

    if jet_branches:
        jets = branches[jet_branches]
    else:
        jets = None

    # delete branches to save memory
    del branches

    # Print some info on dataset. Note: Nr of jets is significantly larger than nr of quark/gluon jets.
    # This is because we only know for sure which jets are quark or gluon jets from the Parton Initiator,
    # which in turn means that we can at most obtain 1 quark or gluon jet per event.
    print(
        f"Number of jets in dataset:\t\t{np.count_nonzero(jets_eta_pt_cut[na.jetpt])}"
    )

    # apply cuts: -2 < eta < 2 and jet_pt > 130 GeV
    if cut:
        print(f"Applying cuts: -{eta_max} < eta < {eta_max} and jet_pt > {pt_min} GeV")
        jets_recur = jets_recur[
            (abs(jets_eta_pt_cut[na.jet_eta]) < eta_max)
            & (jets_eta_pt_cut[na.jetpt] > pt_min)
        ]
        if jet_branches:
            jets = jets[
                (abs(jets_eta_pt_cut[na.jet_eta]) < eta_max)
                & (jets_eta_pt_cut[na.jetpt] > pt_min)
            ]

    # Avoid putting to much in memmory
    del jets_eta_pt_cut

    # apply kt_cut of kt > 1.0 GeV
    if kt_cut:
        print(f"Applying cut: kt > {kt_cut} GeV on all splittings")
        # get kt values
        jets_recur_kt = (
            jets_recur.sigJetRecur_jetpt
            * jets_recur.sigJetRecur_dr12
            * jets_recur.sigJetRecur_z
        )

        # cut kts
        jets_recur = jets_recur[jets_recur_kt > kt_cut]

        if jet_branches:
            jets = jets[jets_recur_kt > kt_cut]

        # hist gluons TODO keep for possible later analysis: histograms
        # g_kts_flat = ak.flatten(ak.flatten(g_jets_kt)).to_list()
        # g_kts_hist = np.histogram(g_kts_flat, bins=range(round(max(g_kts_flat))))

        # hist quarks TODO keep for possible later analysis: histograms
        # q_kts_flat = ak.flatten(ak.flatten(q_jets_kt)).to_list()
        # q_kts_hist = np.histogram(q_kts_flat, bins=range(round(max(q_kts_flat))))
    
    # apply dr_cut
    if dr_cut:
        print(f"Applying cut: dr12 or Rg > {dr_cut} on all splittings")
        jets_recur = jets_recur[jets_recur[na.recur_dr] > dr_cut]
        
        # TODO remove the jet for which initial dr is higher than dr_cut?
        #if jet_branches:
        #    jets = jets[jets_recur[na.recur_dr] > dr_cut]
        
        
    # remove empty additions from recursive jets and flatten them, i.e. take jet out of event nesting
    if jet_branches:
        jets_recur, jets = select_non_empty_branches(
            jets_recur, 
            non_empty_key=jet_recur_branches[0],
            branches_non_recur = jets,
        )
    else:
        jets_recur, _ = select_non_empty_branches(
            jets_recur, 
            non_empty_key=jet_recur_branches[0],
        )

    jets_recur = flatten_array(jets_recur)
    if jet_branches:
        jets = flatten_array(jets)
    
    if cut: # check here since remove empty branches changes counts
        print(f"\tjets left after cuts:\t{len(jets_recur[na.recur_jetpt] > 0)}")

    return jets_recur, jets


def mix_quark_gluon_samples(file_name, jet_branches=None, g_percentage = 90, kt_cut=None, dr_cut=None):
    """Used to create sample sets from an original Pythia file. This sample set is made
    of mixed quarks and gluon jets. Consistency is adjusted by the percentage of gluon
    jets the sample is to be made of."""   
    
    # open file and select jet data and recursive jet data
    data = load_n_filter_data_qg(file_name, jet_branches=jet_branches, kt_cut=kt_cut, dr_cut=dr_cut)

    # calculate g and q bounds
    total = round(len(data[0])/100) # judge total off of about the total number of gluon jets /100
    g_len = total * g_percentage
    q_len = total *(100 - g_percentage) 
    print(f"Mixed sample: {g_len} gluon jets and {q_len} quark jets")
    out_file = f"samples/mixed/{total*100}jets_pct{g_percentage}g{100-g_percentage}q{'_'+str(dr_cut)+'dr_cut' if dr_cut is not None else ''}{'_'+str(kt_cut)+'kt_cut' if kt_cut is not None else ''}.p"

    # load file if file already exists
    try:
        data = torch.load(out_file)
        print(f"Using file found at:\n\t{out_file}")
        return data[0], data[1], out_file
    except FileNotFoundError:
        pass
    
    if jet_branches:
        # print error message in case sd part and sd recur part are not of equal length 
        if (len(data[0]) != len(data[2])) or (len(data[1]) != len(data[3])):
            print("SD and SD recursive not equal for: ", out_file)
        
        # create dataset from bounds as list
        data = [{**sd, **sd_recur} for sd, sd_recur in zip(data[2].to_list(), data[0].to_list())][:g_len] + [{**sd, **sd_recur} for sd, sd_recur in zip(data[3].to_list(), data[1].to_list())][:q_len]
        
        # shuffle list
        random.shuffle(data)
        
        # reform to: (jets_recur, jets)
        data = ak.Array([{key: d[key] for key in data[0] if key not in jet_branches} for d in data]), ak.Array([{ key: d[key] for key in jet_branches } for d in data])
    
    else:
        # add recur to outfile
        out_file += "_recur"
        
        # create dataset from bounds as list
        data = [{**d} for d in data[0].to_list()][:g_len] + [{**d} for d in data[1].to_list()][:q_len] 
        sd = None
        
        # shuffle list
        random.shuffle(data)
        
        # reform to awkward array
        data = ak.Array(data), sd

    # save mixed sample for easy load next time
    torch.save(
        data,
        out_file,
    )
    print(f"Stored mixed sample at:\n\t{out_file}")
    
    return data[0], data[1], out_file


def load_n_filter_data_qg(
    file_name: str,
    tree_name: str = na.tree,
    cut: bool = True,
    kt_cut: float = None,
    dr_cut: float = None,
    eta_max: float = 2.0,
    pt_min: int = 130,
    jet_branches: list = None,
    jet_recur_branches: list = [na.recur_dr, na.recur_jetpt, na.recur_z],
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """Load in dataset from ROOT file of jet data. Subsequently, the jet data will be
    split into sets for quarks and gluons as well as sets with recursive jet data for
    quarks and gluons. Therefore, the provided dataset is required to contain the
    variables: jetpt, jet_eta, jet_phi, jet_M, jet_area, recur_splits & parton_match_id.
    Exact labels are given in 'names.py'.
    Data will be split into a set of quarks and a set of gluon jets.
    
    Maybe obsolete. However, large sample sets might give issues on systems with
    low RAM. Separating quark & gluon jets from the original set and only using these
    could 

    Args:
        file_name (str): File name of ROOT dataset.
        tree_name (str, optional): Name of the TTree object of the ROOT file. Defaults
                to "jetTreeSig".
        cut (bool, optional): Boolean statement whether to apply cuts. Defaults to True.
        kt_cut (float, optional): Functions as boolean statement whether to apply kt_cut
                to splittings by setting a value. Value for kt cut measured in GeV, sets
                a minumum. Defaults to None, i.e. no kt_cut.
        eta_max (float, optional): Value for eta cut. Defaults to 2.0.
        pt_cut (int, optional): Value for pt cut. Defaults to 130.

    Returns:
        Tuple[ak.Array, ak.Array, ak.Array, ak.Array]: Tuple containing all gluon and
                quark datasets.
    """

    # open file and select jet data and recursive jet data
    branches = uproot.open(file_name)[tree_name].arrays()
    jets_eta_pt_cut = branches[[na.jetpt, na.jet_eta, na.parton_match_id,]]
    jets_recur = branches[jet_recur_branches]

    if jet_branches:
        jets = branches[jet_branches]
    else:
        jets = None
        g_jets = None
        q_jets = None

    # delete branches to save memory
    del branches

    # Print some info on dataset. Note: Nr of jets is significantly larger than nr of quark/gluon jets.
    # This is because we only know for sure which jets are quark or gluon jets from the Parton Initiator,
    # which in turn means that we can at most obtain 1 quark or gluon jet per event.
    print(
        f"Number of jets in dataset:\t\t{np.count_nonzero(jets_eta_pt_cut[na.jetpt])}"
    )
    print(
        f"Number of gluon jets in dataset:\t{np.count_nonzero(jets_eta_pt_cut[na.parton_match_id] == 21)}"
    )
    print(
        f"Number of quark jets in dataset:\t{np.count_nonzero(abs(jets_eta_pt_cut[na.parton_match_id]) < 7)}"
    )

    if jet_branches:
        # select quark and gluon jet data
        g_jets = jets[jets_eta_pt_cut[na.parton_match_id] == 21]
        q_jets = jets[abs(jets_eta_pt_cut[na.parton_match_id]) < 7]
        jets = True

    # select recursive quark and gluon jet data
    g_jets_recur = jets_recur[jets_eta_pt_cut[na.parton_match_id] == 21]
    q_jets_recur = jets_recur[abs(jets_eta_pt_cut[na.parton_match_id]) < 7]
    g_jets_eta_pt_cut = jets_eta_pt_cut[jets_eta_pt_cut[na.parton_match_id] == 21]
    q_jets_eta_pt_cut = jets_eta_pt_cut[abs(jets_eta_pt_cut[na.parton_match_id]) < 7]
    del jets_eta_pt_cut
    del jets_recur

    # apply cuts: -2 < eta < 2 and jet_pt > 130 GeV
    if cut:
        print("Applying cuts: -2 < eta < 2 and jet_pt > 130 GeV")
        g_jets_recur = g_jets_recur[
            (abs(g_jets_eta_pt_cut[na.jet_eta]) < eta_max)
            & (g_jets_eta_pt_cut[na.jetpt] > pt_min)
        ]
        q_jets_recur = q_jets_recur[
            (abs(q_jets_eta_pt_cut[na.jet_eta]) < eta_max)
            & (q_jets_eta_pt_cut[na.jetpt] > pt_min)
        ]
        if jet_branches:
            g_jets = g_jets[
                (abs(g_jets_eta_pt_cut[na.jet_eta]) < eta_max)
                & (g_jets_eta_pt_cut[na.jetpt] > pt_min)
            ]
            q_jets = q_jets[
                (abs(q_jets_eta_pt_cut[na.jet_eta]) < eta_max)
                & (q_jets_eta_pt_cut[na.jetpt] > pt_min)
            ]

        # Avoid putting to much in memmory
        del g_jets_eta_pt_cut
        del q_jets_eta_pt_cut

    # apply kt_cut of kt > 1.0 GeV
    if kt_cut:
        print(f"Applying cut: kt > {kt_cut} GeV on all splittings")
        # get kt values
        g_jets_recur_kt = (
            g_jets_recur.sigJetRecur_jetpt
            * g_jets_recur.sigJetRecur_dr12
            * g_jets_recur.sigJetRecur_z
        )
        q_jets_recur_kt = (
            q_jets_recur.sigJetRecur_jetpt
            * q_jets_recur.sigJetRecur_dr12
            * q_jets_recur.sigJetRecur_z
        )

        # cut kts
        g_jets_recur = g_jets_recur[g_jets_recur_kt > kt_cut]
        q_jets_recur = q_jets_recur[g_jets_recur_kt > kt_cut]

        # TODO do not cut from jets? unless no splittings remain in jet? -> already selects non empty branches
        if jet_branches:
            pass # TODO
            # g_jets = g_jets[g_jets_recur_kt > kt_cut]
            # q_jets = q_jets[q_jets_recur_kt > kt_cut]

        print(
            f"\tgluon splittings cut:\t\t{1 - np.count_nonzero(q_jets_recur[q_jets_recur > kt_cut]) / np.count_nonzero(q_jets_recur):.2%}"
        )
        print(
            f"\tquark splittings cut:\t\t{1 - np.count_nonzero(q_jets_recur[q_jets_recur > kt_cut]) / np.count_nonzero(q_jets_recur):.2%}"
        )

        # hist gluons TODO keep for possible later analysis: histograms
        # g_kts_flat = ak.flatten(ak.flatten(g_jets_kt)).to_list()
        # g_kts_hist = np.histogram(g_kts_flat, bins=range(round(max(g_kts_flat))))

        # hist quarks TODO keep for possible later analysis: histograms
        # q_kts_flat = ak.flatten(ak.flatten(q_jets_kt)).to_list()
        # q_kts_hist = np.histogram(q_kts_flat, bins=range(round(max(q_kts_flat))))
    
    # apply dr_cut, or a cut on Rg of each splitting
    if dr_cut:
        print(f"Applying cut: dr12 < {dr_cut} on all splittings")
        g_jets_recur = g_jets_recur[g_jets_recur[na.recur_dr] < dr_cut]
        q_jets_recur = q_jets_recur[q_jets_recur[na.recur_dr] < dr_cut]
        
        # TODO don't do anything? -> already selects non empty branches
        if jet_branches:
            pass # TODO 
            # g_jets = g_jets[g_jets_recur[na.recur_dr] < dr_cut]
            # q_jets = q_jets[q_jets_recur[na.recur_dr] < dr_cut]
            
    # remove empty additions from recursive jets and flatten them, i.e. take jet out of event nesting
    if jet_branches:
        g_jets_recur, g_jets = select_non_empty_branches(
            g_jets_recur, 
            non_empty_key=jet_recur_branches[0],
            branches_non_recur = g_jets,
        )
        q_jets_recur, q_jets = select_non_empty_branches(
            q_jets_recur, 
            non_empty_key=jet_recur_branches[0],
            branches_non_recur = q_jets,
        )
    else:
        g_jets_recur, _ = select_non_empty_branches(
            g_jets_recur, non_empty_key=jet_recur_branches[0]
        )
        q_jets_recur, _ = select_non_empty_branches(
            q_jets_recur, non_empty_key=jet_recur_branches[0]
        ) 

    g_jets_recur = flatten_array(g_jets_recur)
    q_jets_recur = flatten_array(q_jets_recur)
    if jet_branches:
        g_jets = flatten_array(g_jets)
        q_jets = flatten_array(q_jets)
        
        
    if cut: # check here since remove empty branches changes counts
        print(f"\tgluon jets left after cuts:\t{len(g_jets_recur[na.recur_jetpt] > 0)}")
        print(
            f"\tquark jets left after cuts:\t{len(q_jets_recur[na.recur_jetpt] > 0)} "
        )

    return g_jets_recur, q_jets_recur, g_jets, q_jets



def load_trials(job_id, remove_unwanted=True, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    """Function to load and possibly filter trials that have been obtained
    from a training session from either regular_training or hyper_training.

    Args:
        job_id (str): job id for reference
        remove_unwanted (bool, optional): removes unwanted (faulty) trials. Defaults to True.
        device (torch.device, optional): (torch) device currently in use. Defaults to torch.device("cuda")iftorch.cuda.is_available()elsetorch.device("cpu").

    Returns:
        list: list of trials obtained from job
    """    
    
    # load trials
    trials_test_list = torch.load(
        f"storing_results/trials_test_{job_id}.p", map_location=device
    )
    trials = trials_test_list["_trials"]
    
    if remove_unwanted:
        # from run excluded files:
        classifaction_check = CLASSIFICATION_CHECK()
        indices_zero_per_anomaly_nine_flag = classifaction_check.classification_all_nines_test(
            trials=trials
        )
        
        # remove unwanted results:
        track_unwanted = list()
        for i in range(len(trials)):
            if (
                trials[i]["result"]["loss"] == 10
                or trials[i]["result"]["hyper_parameters"]["num_layers"] == 2
                # or trials[i]["result"]["hyper_parameters"]["scaler_id"] == "minmax"
                or i in indices_zero_per_anomaly_nine_flag
            ):
                track_unwanted = track_unwanted + [i]

        trials = [i for j, i in enumerate(trials) if j not in track_unwanted]
    
    return trials