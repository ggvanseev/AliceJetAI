cd $PBS_O_WORKDIR
export PATH=$PBS_O_PATH
pip freeze > req_${PBS_JOBID}.txt
python3 analysis/qp_hyper_training_using_cost_codition_tolga.py

