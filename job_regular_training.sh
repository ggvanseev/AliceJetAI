cd $PBS_O_WORKDIR
export PATH=$PBS_O_PATH
pip freeze > logfiles/req_${PBS_JOBID}.txt
python3 analysis/qp_regular_training.py