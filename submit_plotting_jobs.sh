qsub -l 'walltime=2:00:00' -q gpu-nv -V -o logfiles -j oe job_violin_plots.sh
qsub -l 'walltime=2:00:00' -q gpu-nv -V -o logfiles -j oe job_cost_condition_plots.sh
