qsub -l 'walltime=96:00:00' -q multicore -V -o logfiles -j oe job_hyper_training.sh
