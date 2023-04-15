"""
Used to read some contents of a trial.
"""

from functions.data_loader import load_trials


job_ids_n_trials = {"11120653": 11}


def print_trial(trial):
    print(
        f"\nFrom {'job'+str(trial['jid'])+' - ' if 'jid' in trial else ''}trial {trial['tid']}:"
    )
    for par, val in trial["result"]["hyper_parameters"].items():
        print(f"  {par:12}\t  {val}")
    print(f"with loss: \t\t{trial['result']['loss']}")
    print(f"with final cost:\t{trial['result']['final_cost']}")


for job_id, trial in job_ids_n_trials.items():
    # load trials
    trials = load_trials(job_id, remove_unwanted=False)
    if not trials:
        print(
            f"No succesful trial for job: {job_id}. Try to complete a new training with same settings."
        )
        continue

    try:
        if trial != "all":
            trial = trials[trial]
            print_trial(trial)
        else:
            for trial in trials:
                print_trial(trial)
    except:
        print("Not a good job or trial")
