cd $PBS_O_WORKDIR
export PATH=$PBS_O_PATH
python3 plotting/make_violin_plots.py
